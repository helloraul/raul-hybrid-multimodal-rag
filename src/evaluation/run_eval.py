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
    p.add_argument("--debug", action="store_true", help="Print extra diagnostics.")
    return p.parse_args()

# ---------------- IO helpers ----------------
def open_text_file(path: Path) -> Tuple[Any, str, str]:
    """Return (fh, encoding, delimiter)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        enc = "latin-1"
        text = raw.decode(enc, errors="replace")

    sample = text[:4096]
    for delim in [",", ";", "\t", "|"]:
        if delim in sample:
            delimiter = delim
            break
    else:
        delimiter = ","

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
    doc_path: Path
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

def _split_docrefs(cell: str) -> list[str]:
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
    """Index top-level files. If you want recursion, switch to rglob in the loop."""
    exact = {}        # filename.lower() -> Path
    stems = {}        # stem -> Path
    by_stem_list = [] # [(stem, Path)]
    for p in dataset_dir.iterdir():
        if not p.is_file():
            continue
        key = p.name.lower()
        exact[key] = p
        st = _stem(p.name)
        if st:
            stems[st] = p
            by_stem_list.append((st, p))
    return exact, stems, by_stem_list

def resolve_doc_path(docref_cell: str, dataset_dir: Path) -> Optional[Path]:
    """
    Resolve from:
      - URL(s) / paths / filenames
      - wildcard-ish like "*AAPL*"
      - multiple refs in one cell
    Strategy: exact filename -> filename contains -> fuzzy stem match.
    """
    if not docref_cell:
        return None

    # cache per dataset_dir
    cache_key = str(dataset_dir)
    if not hasattr(resolve_doc_path, "_cache") or getattr(resolve_doc_path, "_cache_key", None) != cache_key:
        exact, stems, by_stem_list = build_dataset_index(dataset_dir)
        resolve_doc_path._cache = (exact, stems, by_stem_list)
        resolve_doc_path._cache_key = cache_key
    else:
        exact, stems, by_stem_list = resolve_doc_path._cache

    for tok in _split_docrefs(docref_cell):
        fname = _extract_filename_from_token(tok)
        norm_fname = _normalize(fname)

        # 1) exact filename (case-insensitive)
        p = exact.get(norm_fname)
        if p:
            return p
        if not norm_fname.endswith(".pdf") and (norm_fname + ".pdf") in exact:
            return exact[norm_fname + ".pdf"]

        # 2) "contains" match on normalized token vs filenames
        norm_tok = _normalize(tok)
        if norm_tok:
            for name_lower, pth in exact.items():
                if norm_tok in name_lower:
                    return pth

        # 3) fuzzy stem
        st = _stem(fname) or _stem(tok)
        if st:
            if st in stems:
                return stems[st]
            keys = [k for k, _ in by_stem_list]
            candidates = difflib.get_close_matches(st, keys, n=1, cutoff=0.78)
            if candidates:
                match = candidates[0]
                return dict(by_stem_list)[match]
            for k, pth in by_stem_list:
                if st in k or k in st:
                    return pth

    return None

# ---------------- Column picking ----------------
def _count_nonempty(col_values: List[str]) -> int:
    cnt = 0
    for v in col_values:
        s = str(v or "").strip()
        if s and s.lower() != "nan":
            cnt += 1
    return cnt

def pick_best_column(headers: List[str], rows: List[List[str]], candidates: List[str]) -> Optional[str]:
    """
    Among the provided candidate header names (case-insensitive),
    return the actual header present in the file with the MOST non-empty values.
    """
    lower_to_actual = {h.lower(): h for h in headers}
    available = [lower_to_actual[c.lower()] for c in candidates if c.lower() in lower_to_actual]
    if not available:
        return None

    # Build index map for quick lookups
    idx = {h: i for i, h in enumerate(headers)}

    best = None
    best_count = -1
    for h in available:
        i = idx[h]
        col_vals = [row[i] if i < len(row) else "" for row in rows[1:]]  # skip header row
        c = _count_nonempty(col_vals)
        if c > best_count:
            best = h
            best_count = c
    return best

def detect_mapping(headers: List[str], rows: List[List[str]]) -> Dict[str, str]:
    """
    Prefer the capitalized headers that actually have data:
      - Question over question
      - Answer over answer
      - Source Docs for docref
    Fall back to lowercase only if the preferred header is missing or empty.
    """
    mapping: Dict[str, str] = {}
    lower = {h.lower(): h for h in headers}
    idx = {h: i for i, h in enumerate(headers)}

    def nonempty_count(h: str) -> int:
        if h not in idx:
            return 0
        col_i = idx[h]
        cnt = 0
        for r in rows[1:]:
            v = (r[col_i] if col_i < len(r) else "").strip()
            if v and v.lower() != "nan":
                cnt += 1
        return cnt

    # ---- question column
    q_pref = lower.get("question")  # actual header for capitalized OR lowercase (they both map here)
    # We want to explicitly check both variants:
    q_cap = "Question" if "Question" in headers else None
    q_low = "question" if "question" in headers else None

    if q_cap and nonempty_count(q_cap) > 0:
        mapping["question"] = q_cap
    elif q_low and nonempty_count(q_low) > 0:
        mapping["question"] = q_low

    # ---- ground_truth / answer column
    a_cap = "Answer" if "Answer" in headers else None
    a_low = "answer" if "answer" in headers else None

    if a_cap and nonempty_count(a_cap) > 0:
        mapping["ground_truth"] = a_cap
    elif a_low and nonempty_count(a_low) > 0:
        mapping["ground_truth"] = a_low

    # ---- docref column
    # Prefer "Source Docs", then common fallbacks
    doc_candidates = [h for h in headers if h.lower() in {
        "source docs", "sourcedocs", "source_docs",
        "doc", "document", "filename", "file", "path"
    }]

    # Prefer exact "Source Docs" if it exists & has data
    if "Source Docs" in doc_candidates and nonempty_count("Source Docs") > 0:
        mapping["docref"] = "Source Docs"
    else:
        # Otherwise pick the first candidate with data
        best = None
        best_cnt = 0
        for h in doc_candidates:
            c = nonempty_count(h)
            if c > best_cnt:
                best, best_cnt = h, c
        if best and best_cnt > 0:
            mapping["docref"] = best
            
        if debug:
            print("📂 Loading questions...")
            print(" header counts:",
            {h: sum(1 for r in rows[1:] if (idx[h] < len(r) and str(r[idx[h]]).strip())) for h in headers})
            print(f"🔎 Detected column mapping: {mapping}")

    return mapping


# ---------------- Loader ----------------
def load_questions_table(qfile: Path, dataset_dir: Path, limit: Optional[int]=None, debug: bool=False) -> List[QAItem]:
    fh, enc, delim = open_text_file(qfile)
    rdr = csv.reader(fh, delimiter=delim)
    rows = list(rdr)
    fh.close()

    if not rows:
        return []

    headers = [h.strip() for h in rows[0]]
    mapping = detect_mapping(headers, rows)
    if debug:
        print("📂 Loading questions...")
        print(f"🔎 Detected column mapping: {mapping}")

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
        print("⚠️  Required columns not found (need at least Question + Source Docs).")
        return []

    items: List[QAItem] = []
    skipped_unresolved = 0
    for i, r in enumerate(rows[1:], start=2):
        q = (r[qi] if qi < len(r) else "").strip()
        dcell = (r[di] if di < len(r) else "").strip()
        gt = (r[gi] if gi is not None and gi < len(r) else "").strip() or None

        if debug and i <= 6:
            print({"question": q[:120], "docref": dcell[:120], "ground_truth": (gt or "")[:120]})

        if not q or not dcell:
            continue

        doc = resolve_doc_path(dcell, dataset_dir)
        if not doc:
            skipped_unresolved += 1
            continue

        items.append(QAItem(
            question=q,
            doc_path=doc,
            ground_truth=gt,
            raw_row={"row_index": i, "docref": dcell}
        ))
        if limit and len(items) >= limit:
            break

    print(f"🧾 Loaded {len(items)} evaluable rows.")
    if skipped_unresolved:
        print(f"⚠️  {skipped_unresolved} rows skipped — could not resolve 'Source Docs' to a local file.")
    return items

# ---------------- Prediction ----------------
def predict_for_item(item: QAItem, pipeline: str) -> str:
    if not HAVE_MAIN:
        return "[eval] src/main.py not importable"

    q = item.question
    d = item.doc_path

    if pipeline == "agentic" and hasattr(app, "agentic_pipeline_wrapper"):
        return app.agentic_pipeline_wrapper(d, q)
    if pipeline == "langgraph" and hasattr(app, "langgraph_pipeline"):
        return app.langgraph_pipeline(d, q)
    if hasattr(app, "manual_pipeline"):
        return app.manual_pipeline(d, q)

    return "[eval] no pipeline available"

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

# ---------------- Main ----------------
def main():
    print(" Current working directory:", os.getcwd())
    print(" Does .env exist?", (PROJECT_ROOT / ".env").exists())

    args = parse_args()
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
        pred = predict_for_item(item, args.pipeline)
        predictions.append({
            "question": item.question,
            "doc_path": str(item.doc_path),
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
    else:
        print("ℹ️  No ground truths found — metrics skipped.")

if __name__ == "__main__":
    main()
