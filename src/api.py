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

# -------------------- helpers from your repo --------------------
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
    """Try your existing extractor; if empty, fall back to pdftotext -layout."""
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

    # ---- 6) LLM call ----
    cq = f"{RAG_INSTRUCTION}\n\nQuestion: {req.question}"
    try:
        answer = (app_main.generate_answer_from_context(context, cq) or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    BAD = ("i'm ready to answer","go ahead and ask","there is no question","how can i assist","i'm here to help")
    if not answer or any(s in answer.lower() for s in BAD):
        answer = "I don't know."

    citations: List[Citation] = [
        Citation(file=f, page=pg + 1, score=sc, kind=kind) for (_t, f, pg, sc, kind) in chosen
    ]
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
            docs,
            per_file_limit=PER_PAGE_CLIP,
            max_docs=min(8, len(docs)),
            window=900
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

# --- tolerant finance probes (replace your KEY_PROBE) ---
KEY_PROBE = [
    r"\bcondensed\W+consolidated\W+(statements?|statement)\W+of\W+operations\b",
    r"\b(statement|statements?)\W+of\W+operations\b",
    r"\b(in|amounts?\s+in)\W+millions\b",
    r"\bnet\W{0,12}sales\b",
    r"\btotal\W{0,12}net\W{0,12}sales\b",
    r"\brevenue\w*\b",
    r"\bsales\W+by\W+(segment|category|geograph\w*)\b",
]

def _keyword_probe_chunks(
    docs: list[Path],
    *,
    per_file_limit: int = 2200,
    max_docs: int = 6,
    window: int = 900
) -> tuple[str, list[Citation]]:
    """Probe doc-level text with tolerant regex; extract windows around each hit."""
    import re
    pat = re.compile("|".join(KEY_PROBE), flags=re.I | re.DOTALL)

    merged: list[str] = []
    cites: list[Citation] = []
    total = 0
    hits_total = 0

    for p in docs[:max_docs]:
        raw = _extract_doc_text_cached(str(p)) or ""
        if not raw:
            continue

        txt = raw.replace("\u2019", "'")
        low = txt.lower()

        windows: list[tuple[int, int]] = []
        for m in pat.finditer(low):
            c = m.start()
            a = max(0, c - window)
            b = min(len(txt), c + window)
            windows.append((a, b))

        # Merge overlapping/nearby windows
        if windows:
            windows.sort()
            merged_spans: list[tuple[int, int]] = []
            ca, cb = windows[0]
            for a, b in windows[1:]:
                if a <= cb + 80:
                    cb = max(cb, b)
                else:
                    merged_spans.append((ca, cb))
                    ca, cb = a, b
            merged_spans.append((ca, cb))

            # Stitch windows up to per_file_limit
            pieces, used = [], 0
            for a, b in merged_spans:
                seg = txt[a:b]
                if used + len(seg) > per_file_limit:
                    seg = seg[: max(0, per_file_limit - used)]
                if seg:
                    pieces.append(seg)
                    used += len(seg)
                if used >= per_file_limit:
                    break
            piece = "\n".join(pieces)
            hits_total += len(merged_spans)
        else:
            # No direct match — try mid-doc slice
            mid = len(txt) // 2
            a = max(0, mid - per_file_limit // 2)
            b = min(len(txt), mid + per_file_limit // 2)
            piece = txt[a:b]

        chunk = f"\n\n===== FILE: {p.name} =====\n{piece}"
        take = chunk if total + len(chunk) <= MAX_CTX else chunk[: MAX_CTX - total]
        merged.append(take)
        # label as 'probe' so we know which path produced the text
        cites.append(Citation(file=p.name, page=0, score=0.0, kind="probe"))
        total += len(take)
        if total >= MAX_CTX:
            break

    logger.debug(f"[ASK] keyword-probe windows={hits_total}, ctx_added={total}")
    return "".join(merged), cites

