# src/agents/agent_controller.py
from __future__ import annotations
from pathlib import Path
import re

# Safe import of your router. This must NOT import this file back.
try:
    from router.doc_router import route_document
except Exception:
    route_document = None  # we’ll fall back to local OCR if router isn’t available

# -------- Minimal local OCR fallback (no external deps except pillow+pytesseract) --------
def _ocr_once(img, config: str) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, lang="eng", config=config).strip()

def _fallback_ocr_best(image_path: Path) -> str:
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except Exception as e:
        return f"[fallback_ocr unavailable: {e}]"

    if not image_path.exists():
        return f"[fallback_ocr: file not found at {image_path}]"

    try:
        img = Image.open(image_path).convert("L")
        img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)

        psms = [11, 6, 7, 4, 3, 12, 13]
        best_text, best_score = "", -1
        for psm in psms:
            cfg = f"--oem 1 --psm {psm} -c user_defined_dpi=300"
            text = _ocr_once(img, cfg)
            score = sum(ch.isalnum() for ch in text)
            if score > best_score:
                best_text, best_score = text, score
        return best_text
    except Exception as e:
        return f"[fallback_ocr error: {e}]"

# -------- Tiny offline extractors so we can answer without LLM --------
_COMPANY_RE = re.compile(r"(?i)\bcompany[:\.\s]+([A-Za-z0-9 .,&'\-]+)")

def _extract_company(text: str) -> str | None:
    if not text:
        return None
    m = _COMPANY_RE.search(text)
    return m.group(1).strip() if m else None

# -------- Public API: agentic_pipeline --------
def agentic_pipeline(doc_path: str, question: str) -> dict:
    """
    Minimal agentic pipeline:
      1) Try router-based text extraction if available.
      2) Fallback to robust OCR.
      3) If the question is about company name, extract via regex.
      4) Otherwise return a compact snippet.
    Returns dict with "category" and "answer".
    """
    path = Path(doc_path)
    text = ""

    # 1) Try your router (if import succeeded)
    if route_document is not None:
        try:
            text = route_document(str(path)) or ""
        except Exception:
            text = ""

    # 2) Fallback OCR if router gave nothing
    if not isinstance(text, str) or len(text.strip()) < 10:
        text = _fallback_ocr_best(path)

    # 3) Heuristic answering
    q_lower = (question or "").lower()

    if "company" in q_lower and "name" in q_lower:
        company = _extract_company(text or "")
        answer = company if company else "I don't know."
        category = "Finance"  # you can improve classification later
        return {"category": category, "answer": answer}

    # generic fallback: short snippet
    snippet = (text or "").strip()
    snippet = snippet.replace("\n", " ").strip()
    if len(snippet) > 300:
        snippet = snippet[:300] + "…"

    return {"category": "General", "answer": snippet or "I don't know."}

__all__ = ["agentic_pipeline"]
