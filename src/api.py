# src/api.py
from __future__ import annotations

import os
import sys
import logging
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel

# -------------------- import path setup --------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]        # /app
SRC_DIR = PROJECT_ROOT / "src"             # /app/src
for p in (PROJECT_ROOT, SRC_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


from evaluation.run_eval import (
    resolve_doc_paths,
    select_relevant_pages,
    _extract_text_any,
)

# LLM wrapper
import main as app_main

# -------------------- config --------------------
MAX_CTX = int(os.getenv("API_MAX_CTX", "12000"))
PER_PAGE_CLIP = int(os.getenv("API_PER_PAGE_CLIP", "2000"))
DEFAULT_MAX_PAGES = int(os.getenv("API_MAX_PAGES", "6"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/SampleDataSet")).resolve()
DISABLE_COMPRESSION = os.getenv("DISABLE_COMPRESSION", "0") in ("1", "true", "TRUE")

LOG_DEBUG = os.getenv("APP_LOG_LEVEL", "INFO").upper() in ("DEBUG", "TRACE")
logger = logging.getLogger("api")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.DEBUG if LOG_DEBUG else logging.INFO)

app = FastAPI(title="Hybrid RAG API", version="0.2.0")

# -------------------- models --------------------
class EvalRequest(BaseModel):
    question: str
    docref: str
    max_pages: Optional[int] = None

class EvalResponse(BaseModel):
    question: str
    doc_paths: List[str]
    context_chars: int
    predicted: str

class AskRequest(BaseModel):
    question: str
    docref: str = "*"
    max_pages: int = 20
    k_bm25: int = 6
    k_vec: int = 6
    use_neo4j: bool = False

class Citation(BaseModel):
    file: str
    page: int  # 0 means doc-level
    score: float
    kind: str  # 'bm25' | 'tfidf' | 'fallback' | 'neo4j'

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    context_chars: int

# -------------------- small utils --------------------
def _tok(s: str) -> str:
    return (s or "").strip()

def _norm(s: str) -> str:
    return (s or "").replace("\u2019", "'").strip()

def _snippet(s: str, n: int = 220) -> str:
    s = " ".join((s or "").split())
    return s[:n] + ("..." if len(s) > n else "")

def _extract_text_best_effort(path: Path) -> str:
    
    try:
        txt = _extract_text_any(path) or ""
    except Exception:
        txt = ""
    if len(txt.strip()) >= 40:
        return txt
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            subprocess.run(["pdftotext", "-layout", str(path), str(out)],
                           check=True, timeout=30)
            return out.read_text(errors="ignore")
    except Exception:
        return txt
    
import re
from collections import OrderedDict

_QUARTER_FROM_FILE = re.compile(r"(20\d{2})\s*Q([1-4])", re.I)
_MONEY = re.compile(r"\$\s*[\d,]+")

def _format_totals_by_period(context: str, citations) -> str:
    """
    Parse 'Total net sales' lines from the stitched context and label them by file-derived period.
    Returns a concise, ordered multiline string.
    """
    # Map file -> pretty period label
    file_period = {}
    for c in citations:
        m = _QUARTER_FROM_FILE.search(c.file)
        if m:
            file_period[c.file] = f"{m.group(1)} Q{m.group(2)}"
        else:
            file_period[c.file] = c.file

    # For each file block, find the nearest 'Total net sales' line and prefer "Three Months Ended"
    blocks = re.split(r"\n=+\s*FILE:\s*(.+?)\s*(?:\[page.*?\])?\s*=+\n", context)
    # split leaves: [preamble, file1, body1, file2, body2, ...]
    out = OrderedDict()
    for i in range(1, len(blocks), 2):
        fname = blocks[i].strip()
        body = blocks[i+1]
        period = file_period.get(fname, fname)

        # Focus on the section around “Three Months Ended”
        # Capture a small window to bias toward quarterly figures.
        snippet = body
        m_three = re.search(r"Three\s+Months\s+Ended(.{0,800})Total\s+net\s+sales(.{0,200})", body, re.I | re.S)
        m_any   = re.search(r"Total\s+net\s+sales(.{0,200})", body, re.I | re.S)

        line = None
        if m_three:
            window = (m_three.group(0) or "")
            # pull money tokens from the window
            monies = _MONEY.findall(window)
            if monies:
                line = f"{period}: " + " · ".join(monies[:4])
        if not line and m_any:
            window = (m_any.group(0) or "")
            monies = _MONEY.findall(window)
            if monies:
                line = f"{period}: " + " · ".join(monies[:4])

        if line:
            out[period] = line

    # Order periods chronologically if possible (by year then Q)
    def _key(p):
        m = re.match(r"(\d{4})\s+Q([1-4])", p)
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)

    lines = [out[k] for k in sorted(out.keys(), key=_key)]
    return "\n".join(lines)

@lru_cache(maxsize=64)
def _extract_doc_text_cached(p: str) -> str:
    """Cache doc-level extraction to avoid repeated PDF parsing (best-effort)."""
    return _extract_text_best_effort(Path(p)) or ""


def _compress_financial_text(txt: str, max_lines: int = 120) -> str:
    if not txt:
        return ""
    if DISABLE_COMPRESSION:
        return txt
    FIN_KEYWORDS = {
        "revenue","sales","net sales","total net sales","gross","margin","opex","operating",
        "expenses","cost","cogs","services","products","quarter","q1","q2","q3","q4",
        "year","yoy","qoq","increase","decrease","percent","percentage","%","$","million",
        "billion","financial statements","condensed","consolidated","income","statement",
        "net","total","sales by","segment","geographic"
    }
    out: List[str] = []
    for raw in txt.splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        low = line.lower()
        has_num = any(ch.isdigit() for ch in low)
        has_kw = any(k in low for k in FIN_KEYWORDS)
        if has_num or has_kw:
            out.append(line[:240] if len(line) > 240 else line)
            if len(out) >= max_lines:
                break
    if len("\n".join(out).strip()) < 300:
        keep = txt.strip().replace("\r", "")
        return keep[:2500] if len(keep) > 2500 else keep
    return "\n".join(out)

def _safe_compress(txt: str, *, max_lines: int = 120, per_page_clip: int = 2000) -> str:
    if not txt:
        return ""
    if DISABLE_COMPRESSION:
        return txt[:per_page_clip]
    comp = _compress_financial_text(txt, max_lines=max_lines)
    if not comp.strip():
        lines = [ln.strip()[:240] for ln in txt.splitlines() if ln and ln.strip()]
        comp = "\n".join(lines[:max_lines])
    if not comp.strip():
        comp = txt
    return comp[:per_page_clip] if len(comp) > per_page_clip else comp

def _build_page_chunk(file: Path, page_idx: int, raw_text: str) -> Tuple[str, str]:
    txt = _safe_compress(raw_text, max_lines=120, per_page_clip=PER_PAGE_CLIP)
    header = f"\n\n===== FILE: {file.name} [page {page_idx + 1}] =====\n"
    return header + txt, f"{file.name} p.{page_idx + 1}"

# Optional TF-IDF rerank
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    if LOG_DEBUG:
        logger.debug("scikit-learn not available; TF-IDF re-rank will be skipped")

def _tfidf_over_chunks(query: str, chunks: List[Tuple[str, str, str]], top_k: int) -> List[Tuple[int, float]]:
    if not _HAS_SK or not chunks:
        return []
    corpus = [c[0] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    X = vec.fit_transform(corpus)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    ranked = sorted([(i, float(sims[i])) for i in range(len(chunks))], key=lambda t: t[1], reverse=True)
    return ranked[: max(1, top_k)]

def _merge_and_clip(chunks_ordered: List[str], max_ctx: int) -> str:
    out, total = [], 0
    for ch in chunks_ordered:
        if not ch:
            continue
        take = ch if total + len(ch) <= max_ctx else ch[: max_ctx - total]
        out.append(take)
        total += len(take)
        if total >= max_ctx:
            break
    return "".join(out)

# -------------------- health --------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/llm-ping")
def llm_ping():
    try:
        out = (app_main.generate_answer_from_context("say OK", "say OK") or "").strip()
        return {"ok": "ok" in out.lower(), "answer": out[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------------------- /eval --------------------
@app.post("/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    q = _tok(req.question)
    if not q:
        raise HTTPException(status_code=400, detail="Missing question.")
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=500, detail=f"DATASET_DIR not found: {DATASET_DIR}")

    docs = resolve_doc_paths(req.docref, DATASET_DIR)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not resolve docref: {req.docref}")

    logger.info(f"[EVAL] q='{q}' docs={len(docs)}")
    max_pages = req.max_pages or DEFAULT_MAX_PAGES
    page_hits: List[Tuple[Path, int, str]] = select_relevant_pages(docs, q, max_pages=max_pages)
    logger.debug(f"[EVAL] page_hits={len(page_hits)}")

    merged_parts: List[str] = []
    total = 0

    def _add_chunk(label: str, body: str):
        nonlocal total
        chunk = f"\n\n===== {label} =====\n{body}"
        if total + len(chunk) > MAX_CTX:
            chunk = chunk[: MAX_CTX - total]
        merged_parts.append(chunk)
        total += len(chunk)

    if page_hits:
        for p, i, page_txt in page_hits:
            raw_txt = (page_txt or "").strip()
            if len(raw_txt) < 80:
                raw_txt = _extract_doc_text_cached(str(p)) or raw_txt
            comp = _safe_compress(raw_txt, max_lines=120, per_page_clip=PER_PAGE_CLIP)
            _add_chunk(f"FILE: {p.name} [page {i+1}]", comp)
            logger.debug(f"[EVAL] add {p.name} p={i+1} chunk={len(comp)} total={total}")
            if total >= MAX_CTX:
                break
    else:
        for p in docs:
            t = _extract_doc_text_cached(str(p))
            comp = _safe_compress(t, max_lines=120, per_page_clip=PER_PAGE_CLIP)
            _add_chunk(f"FILE: {p.name}", comp)
            logger.debug(f"[EVAL] add {p.name} chunk={len(comp)} total={total}")
            if total >= MAX_CTX:
                break

    context = "".join(merged_parts)
    logger.debug(f"[EVAL] ctx_len={len(context)}")

    if not context.strip():
        return EvalResponse(question=q, doc_paths=[str(p) for p in docs], context_chars=0, predicted="I don't know.")

    instruction = (
        "You are a precise financial analyst.\n"
        "Answer the question using ONLY the provided context below.\n"
        "If the context is insufficient, reply exactly: I don't know.\n"
        "Be concise. Include specific figures and trends if present.\n"
        "At the end, cite the source filenames and page numbers you used in parentheses.\n"
        "Do NOT ask the user to restate the question."
    )
    composed_question = f"{instruction}\n\nQuestion: {q}"

    try:
        answer = (app_main.generate_answer_from_context(context, composed_question) or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    BAD = ("i'm ready to answer", "go ahead and ask", "there is no question", "how can i assist", "i'm here to help")
    if not answer or any(s in answer.lower() for s in BAD):
        answer = "I don't know."

    return EvalResponse(
        question=q,
        doc_paths=[str(p) for p in docs],
        context_chars=len(context),
        predicted=answer,
    )


# -------------------- /ask (chat-style) --------------------
RAG_INSTRUCTION = (
    "You are a precise financial analyst. Use ONLY the provided context to answer the question. "
    "If the context is insufficient, reply exactly: \"I don't know.\" "
    "Be concise. Include specific figures and trends when present. "
    "End with citations in parentheses using the filename and page numbers you used."
)
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest = Body(...)):
    # ---- 0) input & dataset checks ----
    q = _norm(req.question)
    if not q:
        raise HTTPException(status_code=400, detail="Missing question.")
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=500, detail=f"DATASET_DIR not found: {DATASET_DIR}")

    docs = resolve_doc_paths(req.docref, DATASET_DIR)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not resolve docref: {req.docref}")

    logger.info(f"[ASK] q='{req.question}' docs={len(docs)} k_bm25={req.k_bm25} k_vec={req.k_vec} max_pages={req.max_pages}")

    # ---- 1) BM25-ish initial page candidates ----
    candidates: List[Tuple[Path, int, str]] = select_relevant_pages(docs, q, max_pages=max(1, req.max_pages))
    logger.debug(f"[ASK] BM25 candidates={len(candidates)}")
    for (p, i, txt) in candidates[: min(len(candidates), req.k_bm25)]:
        logger.debug(f"[ASK] BM25 TOP: {p.name} p={i+1} snip='{_snippet(txt)}'")

    # ---- 1a) If no candidates, try keyword-probe, then whole-doc fallback ----
    if not candidates:
        context, cites = "", []
        try:
            context, cites = _keyword_probe_chunks(
                docs, per_file_limit=PER_PAGE_CLIP, max_docs=min(8, len(docs))
            )
            if context.strip():
                logger.debug(f"[ASK] keyword-probe ctx_len={len(context)}")
            else:
                logger.debug("[ASK] keyword-probe empty → using whole-doc fallback")
        except NameError:
            logger.debug("[ASK] probe helper missing → using whole-doc fallback")

        if not context.strip():
            merged_parts: List[str] = []
            total = 0
            for p in docs:
                raw = _extract_doc_text_cached(str(p)) or ""
                txt = _safe_compress(raw, max_lines=120, per_page_clip=PER_PAGE_CLIP)
                if len(txt) > PER_PAGE_CLIP:
                    txt = txt[:PER_PAGE_CLIP]
                chunk = f"\n\n===== FILE: {p.name} =====\n{txt}"
                if total + len(chunk) > MAX_CTX:
                    chunk = chunk[: MAX_CTX - total]
                merged_parts.append(chunk)
                cites.append(Citation(file=p.name, page=0, score=0.0, kind="fallback"))
                total += len(chunk)
                if total >= MAX_CTX:
                    break
            context = "".join(merged_parts)

        ctx_len = len(context)
        logger.debug(f"[ASK] fallback ctx_len={ctx_len}")

        if not context.strip():
            return AskResponse(question=req.question, answer="I don't know.", citations=[], context_chars=0)

        # --- Heuristic fallback (no LLM) ---
        heur = _extract_total_net_sales(context)
        if heur:
            pretty = _format_totals_by_period(context, cites) or None
            used_tail = "; ".join(sorted({c.file if c.page == 0 else f"{c.file} p.{c.page}" for c in cites}))
            ans = (pretty or heur)
            if used_tail:
                ans = f"{ans} ({used_tail})"
            return AskResponse(
                question=req.question,
                answer=ans,
                citations=cites,
                context_chars=len(context),
            )

        composed_question = f"{RAG_INSTRUCTION}\n\nQuestion: {req.question}"
        try:
            answer = (app_main.generate_answer_from_context(context, composed_question) or "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

        BAD = ("i'm ready to answer", "go ahead and ask", "there is no question", "how can i assist", "i'm here to help")
        if not answer or any(s in answer.lower() for s in BAD):
            answer = "I don't know."

        used = "; ".join(sorted({c.file for c in cites}))
        if used and "(" not in answer[-160:]:
            answer = f"{answer} ({used})"

        return AskResponse(question=req.question, answer=answer, citations=cites, context_chars=ctx_len)


    # ---- 2) Build chunks from BM25 candidates ----
    bm25_chunks: List[Tuple[str, str, str, int]] = []  # (chunk_text, file_name, cite_str, page_idx)
    for p, i, txt in candidates:
        raw_txt = (txt or "").strip()
        if len(raw_txt) < 80:
            raw_txt = _extract_doc_text_cached(str(p)) or raw_txt
        ch, cite_str = _build_page_chunk(p, i, raw_txt)
        bm25_chunks.append((ch, p.name, cite_str, i))

    # ---- 3) TF-IDF over chunks (vector-ish) ----
    tfidf_rank = _tfidf_over_chunks(q, [(c[0], c[1], c[2]) for c in bm25_chunks], top_k=max(1, req.k_vec))
    logger.debug(f"[ASK] TF-IDF hits={len(tfidf_rank)}")
    for (idx, score) in tfidf_rank[: min(len(tfidf_rank), req.k_vec)]:
        ch_text, file_name, _cite_str, page_idx = bm25_chunks[idx]
        logger.debug(f"[ASK] TFIDF TOP: {file_name} p={page_idx+1} score={score:.3f} snip='{_snippet(ch_text)}'")

    # ---- 4) Merge (dedup by file/page) — keep BM25 top-k + remaining TF-IDF ----
    chosen: List[Tuple[str, str, int, float, str]] = []  # (text, file, page_idx, score, kind)
    seen = set()

    for ch_text, file_name, _cite, page_idx in bm25_chunks[: max(1, req.k_bm25)]:
        key = (file_name, page_idx)
        if key in seen:
            continue
        seen.add(key)
        chosen.append((ch_text, file_name, page_idx, 1.0, "bm25"))

    for (idx, score) in tfidf_rank:
        ch_text, file_name, _cite, page_idx = bm25_chunks[idx]
        key = (file_name, page_idx)
        if key in seen:
            continue
        seen.add(key)
        chosen.append((ch_text, file_name, page_idx, float(score), "tfidf"))

    if LOG_DEBUG:
        logger.debug(f"[ASK] chosen chunks={len(chosen)} (pre-clip)")
        for t, f, pg, sc, kind in chosen[:8]:
            logger.debug(f"[ASK] KEEP {kind:6s} {f} p={pg+1} score={sc:.3f} snip='{_snippet(t)}'")

    # ---- 5) Build final context ----
    ordered_texts = [t for (t, *_rest) in chosen]
    context = _merge_and_clip(ordered_texts, MAX_CTX)
    logger.debug(f"[ASK] final context chars={len(context)} from {len(ordered_texts)} chunks")

    if not context.strip():
        return AskResponse(question=req.question, answer="I don't know.", citations=[], context_chars=0)

    # Build citations list NOW so anything below can use it
    citations_preview: List[Citation] = [
        Citation(file=f, page=pg + 1, score=sc, kind=kind) for (_t, f, pg, sc, kind) in chosen
    ]

    # ---- 5.5) Heuristic short-circuit for “Total net sales” (structured) ----
    pretty = _format_sales_answer(context, citations_preview)
    if pretty:
        return AskResponse(
            question=req.question,
            answer=pretty,
            citations=citations_preview,
            context_chars=len(context),
        )

    # Optional heuristic (no LLM) if pretty didn’t trigger
    heur = _extract_total_net_sales(context)
    if heur:
        used_tail = "; ".join(sorted({f"{c.file} p.{c.page}" for c in citations_preview}))
        ans = heur if not used_tail else f"{heur} ({used_tail})"
        return AskResponse(
            question=req.question,
            answer=ans,
            citations=citations_preview,
            context_chars=len(context),
        )

    # ---- 6) LLM call ----
    cq = f"{RAG_INSTRUCTION}\n\nQuestion: {req.question}"
    try:
        answer = (app_main.generate_answer_from_context(context, cq) or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    BAD = ("i'm ready to answer","go ahead and ask","there is no question","how can i assist","i'm here to help")
    if not answer or any(s in answer.lower() for s in BAD):
        answer = "I don't know."

    # Reuse the same citations
    citations = citations_preview
    tail = "; ".join(sorted({f"{c.file} p.{c.page}" for c in citations}))
    if tail and "(" not in answer[-160:]:
        answer = f"{answer} ({tail})"

    return AskResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        context_chars=len(context),
    )

@app.post("/debug/ask-context")
def debug_ask_context(req: AskRequest = Body(...), preview_chars: int = 6000):
    docs = resolve_doc_paths(req.docref, DATASET_DIR)
    q = _tok(req.question)
    candidates = select_relevant_pages(docs, q, max_pages=max(1, req.max_pages))

    merged_parts: List[str] = []
    cites: List[Dict[str, int | str]] = []
    total = 0
    strategy = "bm25"

    if not candidates:
        # 🔎 try probe first
        ctx_probe, probe_cites = _keyword_probe_chunks(
           docs, per_file_limit=PER_PAGE_CLIP,max_docs=min(8, len(docs)),
        )
        if ctx_probe.strip():
            strategy = "probe"
            merged_parts.append(ctx_probe[: MAX_CTX])
            for c in probe_cites:
                cites.append({"file": c.file, "page": c.page, "kind": c.kind})
        else:
            # old whole-doc compressed fallback
            strategy = "whole_doc"
            for p in docs:
                raw = _extract_doc_text_cached(str(p))
                txt = _safe_compress(raw, max_lines=120, per_page_clip=PER_PAGE_CLIP)
                chunk = f"\n\n===== FILE: {p.name} =====\n{txt}"
                if total + len(chunk) > MAX_CTX:
                    chunk = chunk[: MAX_CTX - total]
                merged_parts.append(chunk)
                cites.append({"file": p.name, "page": 0, "kind": "fallback"})
                total += len(chunk)
                if total >= MAX_CTX:
                    break
    else:
        # same as before (BM25 → TF-IDF → chosen chunks)
        ...
        cites.append({"file": file_name, "page": page_idx + 1, "kind": "bm25/tfidf"})

    context = "".join(merged_parts)
    return {
        "strategy": "probe" if not candidates else "bm25+tfidf",
        "context_chars": len(context),
        "context_preview": context[:preview_chars],
        "citations": cites,
    }

import re

KEY_PROBE = [
    r"\btotal\s+net\s+sales\b",
    r"\bnet\s+sales\b",
    r"\brevenue[s]?\b",
    r"\bby\s+(segment|category|geograph\w+)\b",
    r"\bin\s+millions\b",
    r"\b(in|$)\s*USD\b",
]

def _keyword_probe_chunks(
    docs: list[Path],
    per_file_limit: int = 1800,     # a bit larger so we keep more of the table block
    max_docs: int = 6,
    pre_lines: int = 3,
    post_lines: int = 6,
) -> tuple[str, list[Citation]]:
    """
    Scan doc-level text for finance keywords and return stitched context with a
    window of lines around each match so numbers aren't lost.
    """
    merged, cites, total = [], [], 0
    pat = re.compile("|".join(KEY_PROBE), flags=re.I)

    for p in docs[:max_docs]:
        raw = _extract_doc_text_cached(str(p))
        if not raw:
            continue

        # normalize and split into lines
        lines = [ln.rstrip() for ln in raw.splitlines()]
        keep_idxs = set()

        # find all matching lines
        for idx, ln in enumerate(lines):
            if pat.search(ln):
                start = max(0, idx - pre_lines)
                end = min(len(lines), idx + 1 + post_lines)
                keep_idxs.update(range(start, end))

        # build the piece to keep
        if keep_idxs:
            ordered = [lines[i] for i in sorted(keep_idxs)]
            piece = "\n".join(ordered)
        else:
            # fallback: mid-doc slice
            mid = len(raw) // 2
            piece = raw[max(0, mid - per_file_limit // 2): mid + per_file_limit // 2]

        if len(piece) > per_file_limit:
            piece = piece[:per_file_limit]

        chunk = f"\n\n===== FILE: {p.name} =====\n{piece}"
        merged.append(chunk)
        cites.append(Citation(file=p.name, page=0, score=0.0, kind="probe"))
        total += len(chunk)
        if total >= MAX_CTX:
            break

    return "".join(merged), cites

# --- heuristic: pull "Total net sales" lines if present ---
from typing import Optional  # if not already imported
import re  # if not already imported at top

def _extract_total_net_sales(context: str) -> Optional[str]:
    """
    Cheap parser to surface the 'Total net sales' lines (and nearby numbers)
    so you return something useful even if the LLM refuses to answer.
    """
    if not context:
        return None

    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    header = ""
    # Try to capture the period header (e.g., "Three Months Ended ...")
    for ln in lines[:200]:
        if re.search(r"(Three|Nine)\s+Months\s+Ended", ln, re.I):
            header = ln
            break

    totals = []
    for ln in lines:
        if re.search(r"\bTotal\s+net\s+sales\b", ln, re.I):
            totals.append(ln[:300])

    if not totals:
        return None

    out = []
    if header:
        out.append(header)
    out.extend(totals[:8])  # keep it concise
    return "\n".join(out) if out else None



_QUARTER_FROM_FILE = re.compile(r"(20\d{2})\s*Q([1-4])", re.I)
_MONEY = re.compile(r"\$\s*[\d,]+")

def _format_totals_by_period(context: str, citations) -> str:
    file_period = {}
    for c in citations:
        m = _QUARTER_FROM_FILE.search(c.file)
        file_period[c.file] = f"{m.group(1)} Q{m.group(2)}" if m else c.file

    blocks = re.split(r"\n=+\s*FILE:\s*(.+?)\s*=+\n", context)
    out = OrderedDict()

    for i in range(1, len(blocks), 2):
        fname = blocks[i].strip()
        body = blocks[i+1]
        period = file_period.get(fname, fname)

        m_three = re.search(r"Three\s+Months\s+Ended(.{0,800})Total\s+net\s+sales(.{0,200})", body, re.I | re.S)
        m_any   = re.search(r"Total\s+net\s+sales(.{0,200})", body, re.I | re.S)

        line = None
        if m_three:
            monies = _MONEY.findall(m_three.group(0) or "")
            if monies:
                line = f"{period}: " + " · ".join(monies[:4])
        if not line and m_any:
            monies = _MONEY.findall(m_any.group(0) or "")
            if monies:
                line = f"{period}: " + " · ".join(monies[:4])

        if line:
            out[period] = line

    def _key(p):
        m = re.match(r"(\d{4})\s+Q([1-4])", p)
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)

    return "\n".join(out[k] for k in sorted(out.keys(), key=_key))



_DATE_ROW = re.compile(
    r"(?i)\b(three|nine)\s+months\s+ended\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})"
)
# capture “Total net sales … 82,959 … 81,434 … 304,182 … 282,457”
_SALES_LINE = re.compile(
    r"(?i)\btotal\s+net\s+sales\b[^\n]*"
    r"(?:(?:\$|US\$|USD)\s*)?([\d,]+)(?:\s+[\d,]+)?(?:\s+([\d,]+))?(?:\s+([\d,]+))?"
)

_MONTH_TO_Q = {
    "jan": "Q1", "feb": "Q1", "mar": "Q1",
    "apr": "Q2", "may": "Q2", "jun": "Q2",
    "jul": "Q3", "aug": "Q3", "sep": "Q3",
    "oct": "Q4", "nov": "Q4", "dec": "Q4",
}

def _infer_quarter(month_name: str) -> str:
    m = (month_name or "").strip().lower()[:3]
    return _MONTH_TO_Q.get(m, "")

def _extract_periods(context: str):
    """
    Returns a list of (span_start, span_end, label) found in context for
    'Three Months Ended <Month> <DD>, <YYYY>' and 'Nine Months Ended ...'
    """
    out = []
    for m in _DATE_ROW.finditer(context or ""):
        kind = m.group(1).lower()  # three | nine
        mon = m.group(2)
        yr  = m.group(4)
        q = _infer_quarter(mon)
        label = f"{yr} {q} ({'three' if kind=='three' else 'nine'} months)".strip()
        out.append((m.start(), m.end(), label))
    return out

def _closest_period_label(context: str, line_start: int, periods: list) -> str:
    """
    Pick the nearest preceding/nearby period label to the sales line.
    """
    best = ""
    best_dist = 1e9
    for s, e, label in periods:
        if s <= line_start:
            dist = line_start - e
            if 0 <= dist < best_dist and dist < 1200:  # within ~1200 chars above is “nearby”
                best = label
                best_dist = dist
    return best

def _extract_sales_structured(context: str):
    """
    Find 'Total net sales' lines and attach the closest period label.
    Returns list of dicts: {'period': '2022 Q3 (three months)', 'value': '82,959'}
    Handles lines that include multiple numbers (e.g., three vs nine months) by keeping the first.
    """
    results = []
    periods = _extract_periods(context or "")
    for m in _SALES_LINE.finditer(context or ""):
        start = m.start()
        label = _closest_period_label(context, start, periods)
        # first captured group is the main value; others are optional extra columns
        val = next((g for g in m.groups() if g), None)
        if val:
            results.append({"period": label or "Unlabeled period", "value": val.replace(",", "")})
    return results

def _format_sales_answer(context: str, citations) -> str:
    rows = _extract_sales_structured(context)
    if not rows:
        return ""
    # de-dup by (period,value) keep first seen
    seen = set()
    lines = []
    for r in rows:
        key = (r["period"], r["value"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{r['period']}: ${int(r['value']):,}")
    # source tail
    tail = "; ".join(sorted({c.file for c in citations}))
    if tail:
        lines.append(f"({tail})")
    return "\n".join(lines)
