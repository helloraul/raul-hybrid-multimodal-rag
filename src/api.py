# src/api.py
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi import Body


# --- import path setup: add project root AND src ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]        # /app
SRC_DIR = PROJECT_ROOT / "src"             # /app/src
for p in (PROJECT_ROOT, SRC_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Reuse existing helpers
from evaluation.run_eval import (
    resolve_doc_paths,
    select_relevant_pages,
    _extract_text_any,
)

# LLM wrapper (expects generate_answer_from_context(context: str, question: str) -> str)
import main as app_main

# Config
MAX_CTX = int(os.getenv("API_MAX_CTX", "12000"))
PER_PAGE_CLIP = int(os.getenv("API_PER_PAGE_CLIP", "2000"))
DEFAULT_MAX_PAGES = int(os.getenv("API_MAX_PAGES", "6"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/SampleDataSet")).resolve()

app = FastAPI(title="Hybrid RAG API", version="0.1.0")


class EvalRequest(BaseModel):
    question: str
    docref: str
    max_pages: Optional[int] = None


class EvalResponse(BaseModel):
    question: str
    doc_paths: List[str]
    context_chars: int
    predicted: str


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _compress_financial_text(txt: str, max_lines: int = 120) -> str:
    FIN_KEYWORDS = {
        "revenue","sales","net sales","gross","margin","opex","operating",
        "expenses","cost","cogs","services","products","quarter","q1","q2",
        "q3","q4","year","yoy","qoq","increase","decrease","percent","%","$",
        "million","billion"
    }
    if not txt:
        return ""

    out = []
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

    # Fallback: if still too sparse, keep the first N characters of raw text
    if len("\n".join(out).strip()) < 300:
        keep = txt.strip().replace("\r", "")
        if len(keep) > 2500:
            keep = keep[:2500]
        if keep:
            return keep

    return "\n".join(out)


@app.post("/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    # ---------- basic input checks ----------
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing question.")

    if not DATASET_DIR.exists():
        raise HTTPException(status_code=500, detail=f"DATASET_DIR not found: {DATASET_DIR}")

    # ---------- resolve docs ----------
    docs = resolve_doc_paths(req.docref, DATASET_DIR)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not resolve docref: {req.docref}")

    print(f"[EVAL] q={q!r} docref={req.docref!r}")
    print(f"[EVAL] matched {len(docs)} docs")

    # ---------- page selection ----------
    max_pages = req.max_pages or DEFAULT_MAX_PAGES
    page_hits: List[Tuple[Path, int, str]] = select_relevant_pages(docs, q, max_pages=max_pages)
    print(f"[EVAL] page_hits={len(page_hits)}")

    # ---------- build context ----------
    merged_parts: List[str] = []
    total = 0

    def _add_chunk(label: str, body: str) -> None:
        nonlocal total
        chunk = f"\n\n===== {label} =====\n{body}"
        if total + len(chunk) > MAX_CTX:
            chunk = chunk[: MAX_CTX - total]
        merged_parts.append(chunk)
        total += len(chunk)

    if page_hits:
        for p, i, page_txt in page_hits:
            # If page text looks too thin, fall back to whole-doc extract
            raw_txt = (page_txt or "").strip()
            if len(raw_txt) < 80:
                try:
                    raw_txt = _extract_text_best_effort(p) or raw_txt
                except Exception:
                    pass

            # Compress to finance-relevant lines; if still too thin, keep raw slice
            comp = _compress_financial_text(raw_txt)
            if len(comp.strip()) < 300:
                raw_slice = (raw_txt or "").replace("\r", "")
                if len(raw_slice) > PER_PAGE_CLIP:
                    raw_slice = raw_slice[:PER_PAGE_CLIP]
                comp = raw_slice

            if len(comp) > PER_PAGE_CLIP:
                comp = comp[:PER_PAGE_CLIP]

            _add_chunk(f"FILE: {p.name} [page {i+1}]", comp)
            print(f"[EVAL] add page {p.name}#{i+1} (chunk={len(comp)}) total={total}")
            if total >= MAX_CTX:
                break
    else:
        # Fallback: compressed whole-doc extracts
        for p in docs:
            try:
                t = _extract_text_best_effort(p) or ""
            except Exception:
                t = ""
            comp = _compress_financial_text(t)
            if len(comp.strip()) < 300:
                # ensure the LLM has *something*
                keep = (t or "").replace("\r", "")
                comp = keep[:PER_PAGE_CLIP] if len(keep) > PER_PAGE_CLIP else keep
            if len(comp) > PER_PAGE_CLIP:
                comp = comp[:PER_PAGE_CLIP]

            _add_chunk(f"FILE: {p.name}", comp)
            print(f"[EVAL] add doc {p.name} (chunk={len(comp)}) total={total}")
            if total >= MAX_CTX:
                break

    context = "".join(merged_parts)
    if len(context) > MAX_CTX:
        context = context[:MAX_CTX]

    print(f"[EVAL] ctx_len={len(context)}")

    # ---------- if no context, short-circuit ----------
    if not context.strip():
        return EvalResponse(
            question=q,
            doc_paths=[str(p) for p in docs],
            context_chars=0,
            predicted="I don't know."
        )

    # ---------- firm instruction wrapper (guardrail) ----------
    instruction = (
        "You are a precise financial analyst.\n"
        "Answer the question using ONLY the provided context below.\n"
        "If the context is insufficient, reply exactly: I don't know.\n"
        "Be concise. Include specific figures and trends if present.\n"
        "At the end, cite the source filenames and page numbers you used in parentheses.\n"
        "Do NOT mention having no question, and do NOT ask the user to provide a question."
    )
    composed_question = f"{instruction}\n\nQuestion: {q}"

    # ---------- LLM call ----------
    try:
        answer = (app_main.generate_answer_from_context(context, composed_question) or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # ---------- guard against generic/off-task replies ----------
    BAD_SNIPPETS = (
        "i'm ready to answer",
        "go ahead and ask",
        "there is no question",
        "how can i assist",
        "i'm here to help",
        "you haven't actually asked a question",
    )
    if (not answer) or any(s in answer.lower() for s in BAD_SNIPPETS):
        answer = "I don't know."

    return EvalResponse(
        question=q,
        doc_paths=[str(p) for p in docs],
        context_chars=len(context),
        predicted=answer,
    )


# Try to import scikit-learn TF-IDF for vector re-ranking (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Optional Neo4j (safe fallback if not configured)
_NE01_BOLT = os.getenv("NEO4J_BOLT_URI")
try:
    from neo4j import GraphDatabase  # type: ignore
    _HAS_NEO = bool(_NE01_BOLT)
except Exception:
    _HAS_NEO = False

# --- Guardrail instruction we'll reuse ---
RAG_INSTRUCTION = (
    "You are a precise financial analyst.\n"
    "Answer ONLY using the provided context. If the context is insufficient, reply exactly: \"I don't know.\" "
    "Be concise. Include specific figures and trends when present. "
    "End with citations in parentheses using the filename and page numbers you used, e.g. (AAPL_10Q.pdf p.3; p.5)."
)


# ---------- Request/Response models for /ask ----------
class AskRequest(BaseModel):
    question: str
    docref: str = "*"
    # How many candidate pages to pull via BM25-like page selector
    max_pages: int = 20
    # Top-k to keep from BM25 candidate order
    k_bm25: int = 6
    # Top-k to keep from vector re-rank
    k_vec: int = 6
    # Use Neo4j neighborhood expansion (optional)
    use_neo4j: bool = False


class Citation(BaseModel):
    file: str
    page: int
    score: float
    kind: str  # "bm25" | "tfidf" | "neo4j"


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    context_chars: int


# ---------- Small helpers ----------
def _tok(s: str) -> str:
    return (s or "").strip()

def _tfidf_rerank(query: str, items: List[Tuple[Path, int, str]], top_k: int) -> List[Tuple[Path, int, str, float]]:
    """
    Return list of (path, page_index, text, score) by cosine sim over TF-IDF.
    Falls back to empty list if scikit-learn is unavailable or if items is empty.
    """
    if not _HAS_SK or not items:
        return []

    docs = [ _tok(t[2]) for t in items ]
    q = _tok(query)
    # Basic TF-IDF; could switch to char-ngrams for robustness if needed
    vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    X = vec.fit_transform(docs)
    qv = vec.transform([q])
    sims = cosine_similarity(qv, X).ravel()
    ranked_idx = sims.argsort()[::-1][:max(1, top_k)]
    out: List[Tuple[Path, int, str, float]] = []
    for i in ranked_idx:
        pth, pg, txt = items[i]
        out.append((pth, pg, txt, float(sims[i])))
    return out

def _merge_hits(
    bm25_hits: List[Tuple[Path, int, str]],
    tfidf_hits: List[Tuple[Path, int, str, float]],
    k_ctx: int = 8,
) -> Tuple[List[Tuple[Path, int, str]], List[Citation]]:
    """
    Merge BM25 order and TF-IDF order with simple de-dup.
    Keep up to k_ctx unique pages, preserving BM25 order but boosting pages that also appear in TF-IDF top.
    """
    # Map id -> tfidf score
    tf_map: Dict[str, float] = {}
    for p, i, t, sc in tfidf_hits:
        tf_map[f"{p}::{i}"] = sc

    merged: List[Tuple[Path, int, str]] = []
    cites: List[Citation] = []
    seen = set()

    # 1) Walk BM25 and add
    for p, i, txt in bm25_hits:
        key = f"{p}::{i}"
        if key in seen:
            continue
        merged.append((p, i, txt))
        cites.append(Citation(file=p.name, page=i+1, score=tf_map.get(key, 0.0), kind="bm25"))
        seen.add(key)
        if len(merged) >= k_ctx:
            break

    # 2) Fill with remaining TF-IDF
    if len(merged) < k_ctx:
        for p, i, txt, sc in tfidf_hits:
            key = f"{p}::{i}"
            if key in seen:
                # Upgrade citation kind to reflect it also appears in TF-IDF
                for c in cites:
                    if c.file == p.name and c.page == i+1:
                        c.kind = "bm25+tfidf"
                        c.score = max(c.score, sc)
                continue
            merged.append((p, i, txt))
            cites.append(Citation(file=p.name, page=i+1, score=sc, kind="tfidf"))
            seen.add(key)
            if len(merged) >= k_ctx:
                break

    return merged, cites


def _expand_with_neo4j(citations: List[Citation]) -> List[Citation]:
    """
    Optionally expand with Neo4j neighbors. Safe no-op if Neo4j isn’t configured.
    This is a stub that marks we *could* add related pages; customize for your graph schema.
    """
    if not _HAS_NEO:
        return citations

    try:
        driver = GraphDatabase.driver(_NE01_BOLT, auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","")))
        with driver.session() as sess:
            # Example: for each cited file, see if a `RELATED` neighbor exists; add as low-score 'neo4j' hints
            files = list({c.file for c in citations})
            q = """
            UNWIND $files AS f
            MATCH (d:Document {name:f})-[:RELATED]->(n:Document)
            RETURN f AS src, n.name AS neighbor LIMIT 20
            """
            res = sess.run(q, files=files)
            extras = []
            for r in res:
                # we don't have page numbers from graph; use page=0 as a marker
                extras.append(Citation(file=r["neighbor"], page=0, score=0.01, kind="neo4j"))
            # Dedup neighbors already present
            seen = {(c.file, c.page) for c in citations}
            for e in extras:
                if (e.file, e.page) not in seen:
                    citations.append(e)
                    seen.add((e.file, e.page))
    except Exception:
        # Silent fallback if graph isn't reachable or schema doesn't match
        return citations

    return citations

# -------------- /ask endpoint --------------
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest = Body(...)):
    # --- 0) Input + dataset checks ---
    q = _tok(req.question)
    if not q:
        raise HTTPException(status_code=400, detail="Missing question.")

    if not DATASET_DIR.exists():
        raise HTTPException(status_code=500, detail=f"DATASET_DIR not found: {DATASET_DIR}")

    # Resolve candidate doc set from the glob/ref
    docs = resolve_doc_paths(req.docref, DATASET_DIR)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not resolve docref: {req.docref}")

    # --- 1) BM25-like initial page selection ---
    max_pages = max(1, req.max_pages or DEFAULT_MAX_PAGES)
    candidates: List[Tuple[Path, int, str]] = select_relevant_pages(docs, q, max_pages=max_pages)
    print(f"[ASK] q='{q[:80]}' docs={len(docs)} candidates={len(candidates)}")

    # --- 1a) Optional Neo4j expansion (if available + requested) ---
    if getattr(req, "use_neo4j", False) and "neo4j" in globals() and callable(globals().get("_expand_with_neo4j", None)):
        try:
            neo_expanded = _expand_with_neo4j(candidates, limit=max_pages)
            if neo_expanded:
                print(f"[ASK] neo4j expanded +{len(neo_expanded) - len(candidates)}")
                candidates = neo_expanded
        except Exception as e:
            print(f"[ASK] neo4j expansion skipped: {e}")

    # --- Fallback path: if BM25 yielded nothing, build whole-doc context like /eval ---
    if not candidates:
        merged_parts: List[str] = []
        cites: List[Citation] = []
        total = 0

        for p in docs:
            raw = _extract_text_best_effort(p) or ""
            txt = _compress_financial_text(raw)
            if len(txt) > PER_PAGE_CLIP:
                txt = txt[:PER_PAGE_CLIP]

            chunk = f"\n\n===== FILE: {p.name} =====\n{txt}"
            # Enforce MAX_CTX
            if total + len(chunk) > MAX_CTX:
                chunk = chunk[: MAX_CTX - total]

            merged_parts.append(chunk)
            cites.append(Citation(file=p.name, page=0, score=0.0, kind="fallback"))
            total += len(chunk)
            if total >= MAX_CTX:
                break

        context = "".join(merged_parts)
        ctx_len = len(context)
        print(f"[ASK] fallback ctx_len={ctx_len}")

        if not context.strip():
            return AskResponse(question=q, answer="I don't know.", citations=[], context_chars=0)

        # Guarded prompt
        composed_question = f"{RAG_INSTRUCTION}\n\nQuestion: {q}"
        try:
            answer = (app_main.generate_answer_from_context(context, composed_question) or "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

        # Guardrail vs generic/off-task replies
        BAD_SNIPPETS = (
            "i'm ready to answer",
            "go ahead and ask",
            "there is no question",
            "how can i assist",
            "i'm here to help",
        )
        if not answer or any(s in answer.lower() for s in BAD_SNIPPETS):
            answer = "I don't know."

        # Add simple filename citations if not already present
        used = "; ".join(sorted({c.file for c in cites}))
        if used and "(" not in answer[-160:]:
            answer = f"{answer} ({used})"

        return AskResponse(question=q, answer=answer, citations=cites, context_chars=ctx_len)

    # --- 2) Vector-ish re-rank (TF-IDF proxy) over candidates ---
    tfidf_ranked = _tfidf_rerank(q, candidates, top_k=max(1, getattr(req, "k_vec", 6)))

    # --- 3) Merge BM25 + TF-IDF views and keep a compact context set ---
    k_bm25 = max(1, getattr(req, "k_bm25", 6))
    k_ctx = min(DEFAULT_MAX_PAGES, max(k_bm25, getattr(req, "k_vec", 6)))

    context_pages, citations = _merge_hits(
        bm25_hits=candidates[:k_bm25],
        tfidf_hits=tfidf_ranked,
        k_ctx=k_ctx,
    )

    # --- 4) Build context from page-level hits (with doc-level fallback per page) ---
    merged_parts: List[str] = []
    total = 0
    for p, i, raw in context_pages:
        raw_txt = (raw or "").strip()
        if len(raw_txt) < 80:
            try:
                raw_txt = _extract_text_any(p) or raw_txt
            except Exception:
                pass

        txt = _compress_financial_text(raw_txt)
        if len(txt) > PER_PAGE_CLIP:
            txt = txt[:PER_PAGE_CLIP]

        chunk = f"\n\n===== FILE: {p.name} [page {i+1}] =====\n{txt}"
        if total + len(chunk) > MAX_CTX:
            chunk = chunk[: MAX_CTX - total]

        merged_parts.append(chunk)
        total += len(chunk)
        if total >= MAX_CTX:
            break

    context = "".join(merged_parts)
    ctx_len = len(context)
    print(f"[ASK] ctx_len={ctx_len} pages_used={len(context_pages)}")

    if not context.strip():
        return AskResponse(question=q, answer="I don't know.", citations=[], context_chars=0)

    # --- 5) Guarded LLM call ---
    composed_question = f"{RAG_INSTRUCTION}\n\nQuestion: {q}"
    try:
        answer = (app_main.generate_answer_from_context(context, composed_question) or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    BAD_SNIPPETS = (
        "i'm ready to answer",
        "go ahead and ask",
        "there is no question",
        "how can i assist",
        "i'm here to help",
    )
    if not answer or any(s in answer.lower() for s in BAD_SNIPPETS):
        answer = "I don't know."

    # Append concise citations (filenames + page numbers) if the answer doesn't already contain a citation tail
    cite_tail = "; ".join(
        sorted({f"{c.file}[p.{c.page+1}]" if c.page else f"{c.file}" for c in citations})
    )
    if cite_tail and "(" not in answer[-160:]:
        answer = f"{answer} ({cite_tail})"

    return AskResponse(
        question=q,
        answer=answer,
        citations=citations,
        context_chars=ctx_len,
    )

import subprocess
import tempfile

def _extract_text_best_effort(path: Path) -> str:
    """Try your existing extractor; if it comes back empty, fall back to pdftotext."""
    txt = ""
    try:
        txt = _extract_text_any(path) or ""
    except Exception:
        txt = ""

    if len(txt.strip()) >= 40:
        return txt

    # Fallback: poppler's pdftotext (we installed poppler-utils in the Dockerfile)
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.txt"
            # -layout keeps table-ish spacing, helps the LLM
            subprocess.run(
                ["pdftotext", "-layout", str(path), str(out)],
                check=True, timeout=30
            )
            return out.read_text(errors="ignore")
    except Exception:
        return txt  # give back whatever we had

from fastapi import Query

@app.get("/debug/extract")
def debug_extract(file: str = Query(..., description="filename in SampleDataSet")):
    target = DATASET_DIR / file
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {target}")
    txt = _extract_text_best_effort(target) or ""
    return {
        "file": file,
        "chars": len(txt),
        "preview": (txt[:800] if txt else "")
    }
