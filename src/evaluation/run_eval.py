#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation runner:
- Loads questions from SampleDataSet/SampleQuestions/... (or --questions)
- Resolves document paths from 'Source Docs' (handles wildcards like *AAPL*)
- Calls pipeline wrappers in src/main.py (manual/langgraph/agentic)
- Reports Token-F1 (more forgiving than exact match)

Usage:
  python src/evaluation/run_eval.py
  python src/evaluation/run_eval.py --limit 25 --debug
  python src/evaluation/run_eval.py --pipeline agentic
  python src/evaluation/run_eval.py --questions SampleDataSet/SampleQuestions/questions_with_partial_answers.csv
"""

from __future__ import annotations
import os, sys, csv, json, re, unicodedata, difflib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

# ---------------- Path & env ----------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # repo root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

# Import your pipelines from src/main.py (main() won't run because of __name__ guard)
try:
    import main as app
    HAVE_MAIN = True
except Exception as e:
    print(f"⚠️  Could not import src/main.py as module: {e}")
    HAVE_MAIN = False

# ---------------- CLI ----------------
import argparse
def parse_args():
    p = argparse.ArgumentParser(description="Run evaluation over SampleDataSet Q&A.")
    p.add_argument("--dataset", type=str, default="SampleDataSet",
                   help="Path to dataset dir (default: SampleDataSet at repo root).")
    p.add_argument("--questions", type=str, default=None,
                   help="Path to questions CSV. If omitted, auto-detects in SampleQuestions/.")
    p.add_argument("--pipeline", type=str, default="agentic",
                   choices=["agentic", "langgraph", "manual"],
                   help="Which pipeline to use.")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    p.add_argument("--max-pages", type=int, default=6, help="Max relevant pages to select across docs.")
    p.add_argument("--debug", action="store_true", help="Print extra diagnostics.")
    return p.parse_args()

# ---------------- IO helpers ----------------
def open_text_file(path: Path) -> Tuple[Any, str, str]:
    """Return (fh, encoding, delimiter)."""
    raw = path.read_bytes()
    enc = "utf-8"
    for enc_try in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc_try)
            enc = enc_try
            break
        except UnicodeDecodeError:
            continue
    else:
        enc = "latin-1"
        text = raw.decode(enc, errors="replace")

    sample = text[:4096]
    delimiter = ","
    for delim in [",", ";", "\t", "|"]:
        if delim in sample:
            delimiter = delim
            break

    return open(path, "r", encoding=enc, newline=""), enc, delimiter

def default_questions_file(dataset_dir: Path) -> Optional[Path]:
    for c in [
        dataset_dir / "SampleQuestions" / "questions_with_partial_answers.csv",
        dataset_dir / "SampleQuestions" / "questions.csv",
        dataset_dir / "questions_with_partial_answers.csv",
        dataset_dir / "questions.csv",
    ]:
        if c.exists():
            return c
    return None

# ---------------- Row model ----------------
@dataclass
class QAItem:
    question: str
    doc_paths: List[Path]          # multiple docs supported
    ground_truth: Optional[str] = None
    raw_row: Dict[str, Any] = None

# ---------------- Normalization helpers ----------------
def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip().strip('"\'')
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("%20", " ")
    s = " ".join(s.lower().split())
    return s

def _stem(s: str) -> str:
    """Lowercase, diacritics stripped, alnum-only of basename without ext."""
    name = Path(s).stem
    return "".join(ch for ch in _normalize(name) if ch.isalnum())

def _split_docrefs(cell: str) -> List[str]:
    if not cell:
        return []
    buff = cell.replace("\r", "\n")
    for sep in [";", "|", "\n", ","]:
        if sep in buff:
            parts = [p.strip() for p in buff.split(sep)]
            break
    else:
        parts = [cell.strip()]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def _extract_filename_from_token(token: str) -> str:
    t = (token or "").strip().strip('"\'')
    # Strip wildcard stars like *AAPL*
    t = t.replace("*", "").strip()
    # URL? take last segment
    try:
        u = urlparse(t)
        if u.scheme and u.netloc:
            return unquote(Path(u.path).name)
    except Exception:
        pass
    return unquote(Path(t).name)

# ---------------- Dataset index & resolver ----------------
def build_dataset_index(dataset_dir: Path):
    """Index files recursively under dataset_dir."""
    exact: Dict[str, Path] = {}        # filename.lower() -> Path
    stems: Dict[str, Path] = {}        # stem -> Path
    by_stem_list: List[Tuple[str, Path]] = [] # [(stem, Path)]
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        key = p.name.lower()
        exact[key] = p
        st = _stem(p.name)
        if st:
            stems[st] = p
            by_stem_list.append((st, p))
    return exact, stems, by_stem_list

def resolve_doc_paths(docref_cell: str, dataset_dir: Path) -> List[Path]:
    """
    Resolve from:
      - URL(s) / paths / filenames
      - wildcard-ish like "*AAPL*"
      - multiple refs in one cell
    Strategy: exact filename -> filename contains -> fuzzy stem match.
    Returns a list of hits (deduped, order by discovery).
    """
    if not docref_cell:
        return []

    # cache index
    cache_key = str(dataset_dir)
    if not hasattr(resolve_doc_paths, "_cache") or getattr(resolve_doc_paths, "_cache_key", None) != cache_key:
        exact, stems, by_stem_list = build_dataset_index(dataset_dir)
        resolve_doc_paths._cache = (exact, stems, by_stem_list)
        resolve_doc_paths._cache_key = cache_key
    else:
        exact, stems, by_stem_list = resolve_doc_paths._cache

    hits: List[Path] = []
    def _add(p: Path):
        if p not in hits:
            hits.append(p)

    for tok in _split_docrefs(docref_cell):
        tok = tok.strip().strip('"\'')
        fname = _extract_filename_from_token(tok)
        norm_fname = _normalize(fname)

        # exact file
        if norm_fname in exact:
            _add(exact[norm_fname])
            continue
        if not norm_fname.endswith(".pdf") and (norm_fname + ".pdf") in exact:
            _add(exact[norm_fname + ".pdf"])
            continue

        # contains (useful for "*AAPL*")
        norm_tok = _normalize(tok).replace("*", "")
        if norm_tok:
            for name_lower, pth in exact.items():
                if norm_tok in name_lower:
                    _add(pth)

        # fuzzy by stem
        st = _stem(fname) or _stem(tok.replace("*",""))
        if st:
            if st in stems:
                _add(stems[st])
            else:
                keys = [k for k, _ in by_stem_list]
                for cand in difflib.get_close_matches(st, keys, n=4, cutoff=0.78):
                    _add(dict(by_stem_list)[cand])
                for k, pth in by_stem_list:
                    if st in k or k in st:
                        _add(pth)

    return hits

# ---------------- Column picking ----------------
def detect_mapping(headers: List[str], rows: List[List[str]], debug: bool = False) -> Dict[str, str]:
    """
    Decide which columns are question / ground_truth / docref by:
      - Preferring capitalized headers if they actually have data
      - Falling back to lowercase variants
      - For docref, prefer 'Source Docs' then common fallbacks with the most data
    """
    mapping: Dict[str, str] = {}
    idx = {h: i for i, h in enumerate(headers)}

    def nonempty_count(h: str) -> int:
        if h not in idx:
            return 0
        i = idx[h]
        cnt = 0
        for r in rows[1:]:
            v = (r[i] if i < len(r) else "")
            v = (v or "").strip()
            if v and v.lower() != "nan":
                cnt += 1
        return cnt

    # ---- question column
    q_cap = "Question" if "Question" in headers else None
    q_low = "question" if "question" in headers else None
    if q_cap and nonempty_count(q_cap) > 0:
        mapping["question"] = q_cap
    elif q_low and nonempty_count(q_low) > 0:
        mapping["question"] = q_low

    # ---- ground_truth / answer column (optional)
    a_cap = "Answer" if "Answer" in headers else None
    a_low = "answer" if "answer" in headers else None
    if a_cap and nonempty_count(a_cap) > 0:
        mapping["ground_truth"] = a_cap
    elif a_low and nonempty_count(a_low) > 0:
        mapping["ground_truth"] = a_low

    # ---- docref column
    candidates = [h for h in headers if h.lower() in {
        "source docs", "sourcedocs", "source_docs",
        "doc", "document", "filename", "file", "path"
    }]
    if "Source Docs" in candidates and nonempty_count("Source Docs") > 0:
        mapping["docref"] = "Source Docs"
    else:
        best, best_cnt = None, 0
        for h in candidates:
            c = nonempty_count(h)
            if c > best_cnt:
                best, best_cnt = h, c
        if best_cnt > 0:
            mapping["docref"] = best

    if debug:
        print("📂 Loading questions...")
        print(" header counts:", {h: nonempty_count(h) for h in headers})
        print(f"🔎 Detected column mapping: {mapping}")

    return mapping

def _digit_density(txt: str) -> float:
    if not txt:
        return 0.0
    digits = sum(ch.isdigit() for ch in txt)
    return digits / max(1, len(txt))

FIN_KEYWORDS = {"revenue","sales","net sales","gross","margin","opex","operating","expenses","cost","cogs","services","products","quarter","q1","q2","q3","q4","year","yoy","qoq","increase","decrease","percent","%","$","million","billion"}

def compress_financial_text(txt: str, max_lines: int = 80) -> str:
    """
    Keep only lines likely to matter: those with digits or finance-y words.
    Also trims super-long lines.
    """
    if not txt:
        return ""
    out = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        has_num = any(ch.isdigit() for ch in low)
        has_kw = any(k in low for k in FIN_KEYWORDS)
        if has_num or has_kw:
            if len(line) > 240:
                line = line[:240]  # hard trim long narrative
            out.append(line)
            if len(out) >= max_lines:
                break
    if not out:
        # fallback: first N short lines
        for raw in txt.splitlines():
            line = raw.strip()
            if line:
                out.append(line[:200])
                if len(out) >= max_lines:
                    break
    return "\n".join(out)

# ---------------- Relevance selection ----------------
STOPWORDS = set("the a an of to in on for and or as at by from with into over between about during before after above below up down out off under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very".split())

def _keywords(text: str) -> set:
    text = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return {w for w in text.split() if len(w) > 2 and w not in STOPWORDS}

def select_relevant_pages(paths: List[Path], question: str, max_pages: int = 12) -> List[Tuple[Path, int, str]]:
    """
    Return up to max_pages (path, page_index, page_text) scored by keyword overlap with the question.
    Requires PyMuPDF (fitz). Falls back to first pages if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    qk = _keywords(question)
    candidates: List[Tuple[float, Path, int, str]] = []  # (score, path, page_i, text)
    for p in paths:
        try:
            doc = fitz.open(p)
        except Exception:
            continue
        for i in range(len(doc)):
            try:
                txt = doc[i].get_text("text")
            except Exception:
                txt = ""
            if not txt:
                continue
            pk = _keywords(txt)
            overlap = len(qk & pk)
            # base overlap normalized by question size
            score = overlap / (len(qk) + 1)
            # boost numeric-heavy pages (tables/figures)
            score += 0.35 * _digit_density(txt)
            if score > 0:
                candidates.append((score, p, i, txt))

        doc.close()

    # If nothing scored, return first pages across docs
    if not candidates:
        out: List[Tuple[Path, int, str]] = []
        for p in paths:
            try:
                doc = fitz.open(p)
                for i in range(min(len(doc), max_pages - len(out))):
                    out.append((p, i, doc[i].get_text("text") or ""))
                doc.close()
            except Exception:
                pass
            if len(out) >= max_pages:
                break
        return out[:max_pages]

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [(p, i, txt) for _, p, i, txt in candidates[:max_pages]]

# ---------------- Prediction ----------------
MAX_CTX = 8000        # much smaller; avoid Groq 413
PER_PAGE_CLIP = 1_200   # shorter slices per page

DEBUG = False           # set in main() from --debug

def _extract_text_any(path: Path) -> str:
    # use the helper you added in main.py (PyMuPDF -> PyPDF2 fallback)
    try:
        from main import _extract_text_any as _ext  # reuse
        return _ext(path)
    except Exception:
        return ""

def predict_for_item(item: QAItem, pipeline: str, max_pages: int) -> str:
    if not HAVE_MAIN:
        return "[eval] src/main.py not importable"

    # 1) Try pipeline wrapper if it accepts a single path
    if pipeline == "agentic" and hasattr(app, "agentic_pipeline_wrapper") and len(item.doc_paths) == 1:
        return app.agentic_pipeline_wrapper(item.doc_paths[0], item.question)

    # 2) Build merged context from *relevant pages* then use your RAG answer
    merged = []
    total = 0

    page_hits = select_relevant_pages(item.doc_paths, item.question, max_pages=max_pages)

    if page_hits:
        for p, i, txt in page_hits:
            txt = compress_financial_text(txt)
            if len(txt) > PER_PAGE_CLIP:
                txt = txt[:PER_PAGE_CLIP]
            chunk = f"\n\n===== FILE: {p.name} [page {i+1}] =====\n{txt}"
            if total + len(chunk) > MAX_CTX:
                chunk = chunk[: MAX_CTX - total]
            merged.append(chunk)
            total += len(chunk)
            if total >= MAX_CTX:
                break
    else:
        # last resort: whole-doc extraction (already in your helper)
        for p in item.doc_paths:
            t = _extract_text_any(p) or ""
            if not t:
                continue
            if len(t) > PER_PAGE_CLIP:
                t = t[:PER_PAGE_CLIP]
            chunk = f"\n\n===== FILE: {p.name} =====\n{t}"
            if total + len(chunk) > MAX_CTX:
                chunk = chunk[: MAX_CTX - total]
            merged.append(chunk)
            total += len(chunk)
            if total >= MAX_CTX:
                break

    context = "".join(merged)
    # ultra-safety clip to avoid provider-side payload errors
    if len(context) > MAX_CTX:
        context = context[:MAX_CTX]

    # If context is > ~6k chars, don't even try Groq; go straight to OpenAI.
    prefer_openai = len(context) > 6_000


    if DEBUG:
        print(f"🧠 Context chars: {len(context):,} across {len(item.doc_paths)} files")

    if (getattr(app, "openai_api_key", None) or getattr(app, "groq_api_key", None)) and context:
        try:
            if getattr(app, "groq_api_key", None) and not prefer_openai:
                 return (app.generate_answer_from_context(context, item.question) or "").strip()
            else:
            # temporarily disable groq path if your app helper always tries Groq first
            # If you can’t control that inside main.py, just fall through to except and let OpenAI path run.
                return (app.generate_answer_from_context(context, item.question) or "").strip()
        except Exception:
             pass


    # last resort
    return (context[:400] or "I don't know.")

# ---------------- Metrics (Token-F1) ----------------
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t]

def _f1(pred: str, gold: str) -> float:
    p = _tok(pred)
    g = _tok(gold)
    if not p or not g:
        return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

# ---------------- Loader ----------------
def load_questions_table(qfile: Path, dataset_dir: Path, limit: Optional[int]=None, debug: bool=False) -> List[QAItem]:
    fh, enc, delim = open_text_file(qfile)
    rdr = csv.reader(fh, delimiter=delim)
    rows = list(rdr)
    fh.close()

    if not rows:
        return []

    headers = [h.strip() for h in rows[0]]
    mapping = detect_mapping(headers, rows, debug=debug)

    # Build header -> position index
    col_idx = {h: i for i, h in enumerate(headers)}
    def col(name: str) -> Optional[int]:
        if name in mapping:
            return col_idx.get(mapping[name])
        return None

    qi = col("question")
    di = col("docref")
    gi = col("ground_truth")

    if qi is None or di is None:
        print("⚠️  Required columns not found (need at least Question + Source Docs/docref).")
        if debug:
            print("Headers:", headers)
            print("Mapping:", mapping)
        return []

    items: List[QAItem] = []
    skipped_unresolved = 0
    for i, r in enumerate(rows[1:], start=2):
        q = (r[qi] if qi is not None and qi < len(r) else "").strip()
        dcell = (r[di] if di is not None and di < len(r) else "").strip()
        gt = (r[gi] if gi is not None and gi < len(r) else "").strip() or None

        if debug and i <= 6:
            print({"question": q[:120], "docref": dcell[:120], "ground_truth": (gt or "")[:120]})

        if not q or not dcell:
            continue

        docs = resolve_doc_paths(dcell, dataset_dir)
        if not docs:
            skipped_unresolved += 1
            continue

        items.append(QAItem(
            question=q,
            doc_paths=docs,
            ground_truth=gt,
            raw_row={"row_index": i, "docref": dcell}
        ))
        if limit and len(items) >= limit:
            break

    print(f"🧾 Loaded {len(items)} evaluable rows.")
    if skipped_unresolved:
        print(f"⚠️  {skipped_unresolved} rows skipped — could not resolve 'Source Docs' to a local file.")
    return items

# ---------------- Main ----------------
def main():
    global DEBUG
    print(" Current working directory:", os.getcwd())
    print(" Does .env exist?", (PROJECT_ROOT / ".env").exists())

    args = parse_args()
    DEBUG = bool(args.debug)

    dataset_dir = (PROJECT_ROOT / args.dataset).resolve()
    if not dataset_dir.exists():
        print(f"❌ Dataset dir not found: {dataset_dir}")
        return

    qfile = Path(args.questions).resolve() if args.questions else default_questions_file(dataset_dir)
    if not qfile or not qfile.exists():
        print("❌ Could not locate questions file. Provide --questions or put it under SampleDataSet/SampleQuestions/.")
        return

    items = load_questions_table(qfile, dataset_dir, limit=args.limit, debug=args.debug)

    predictions = []
    for item in items:
        pred = predict_for_item(item, args.pipeline, max_pages=args.max_pages)
        predictions.append({
            "question": item.question,
            "doc_paths": [str(p) for p in item.doc_paths],
            "predicted": pred,
            "ground_truth": item.ground_truth,
            "raw": item.raw_row,
        })

    out = PROJECT_ROOT / "predictions.json"
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote predictions to {out}")

    gts = [p for p in predictions if p.get("ground_truth")]
    if gts:
        scores = [_f1(p.get("predicted",""), p.get("ground_truth","")) for p in gts]
        avg_f1 = sum(scores) / len(scores) if scores else 0.0
        print(f"📈 Token-F1 on answered rows: {avg_f1:.3f} (n={len(scores)})")
        if args.debug:
            print("— Per-row F1 —")
            for p in gts[:10]:
                s = _f1(p.get("predicted",""), p.get("ground_truth",""))
                print(f" • {s:.3f} :: {p['question'][:70]}...")
    else:
        print("ℹ️  No ground truths found — metrics skipped.")

if __name__ == "__main__":
    main()
