import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- Project root & import path ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

# ---------- Diagnostics ----------
print(" Current working directory:", os.getcwd())
print(" Does .env exist?", (PROJECT_ROOT / ".env").exists())

# ---------- .env ----------
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key  = os.getenv("GROQ_API_KEY")

if not openai_api_key:
    print("⚠️  OPENAI_API_KEY not found — LLM-based answers may be skipped.")
if not groq_api_key:
    print("⚠️  GROQ_API_KEY not found — LangGraph steps using Groq may be skipped.")

# ---------- Local imports ----------
from router.doc_router import route_document
from src.graph.orchestration import build_rag_workflow
from agents.agent_controller import agentic_pipeline
from agents.rag_orchestrator import generate_answer_from_context

# ---------- OCR helpers ----------
def ensure_demo_invoice() -> Path:
    demo_path = PROJECT_ROOT / "examples" / "demo_invoice.png"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    if demo_path.exists():
        return demo_path
    try:
        from PIL import Image, ImageDraw
        W, H = 1400, 900
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        title = "INVOICE"
        lines = [
            "Company: Acme Robotics Inc",
            "Address: 123 Market St, San Francisco, CA 94103",
            "Invoice #: INV-2025-081",
            "Date: 2025-08-07",
            "",
            "Bill To: ABC LLC",
            "Item                 Qty    Unit Price    Amount",
            "GPU Cluster Hours     120       $4.20     $504.00",
            "Storage (TB-Mo)        10      $12.00     $120.00",
            "Support (Premium)       1      $99.00      $99.00",
            "",
            "Subtotal:                                    $723.00",
            "Tax (8.5%):                                  $61.46",
            "Total:                                       $784.46",
        ]
        d.text((W//2 - 120, 40), title, fill="black")
        y = 120
        for line in lines:
            d.text((80, y), line, fill="black")
            y += 48
        img.save(demo_path)
    except Exception as e:
        print(f"⚠️ Could not create demo invoice image: {e}")
    return demo_path

def _ocr_once(img, config: str) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, lang="eng", config=config).strip()

def fallback_ocr_best(image_path: Path) -> str:
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except Exception as e:
        return f"[fallback_ocr unavailable: {e}]"
    if not image_path.exists():
        return f"[fallback_ocr: file not found at {image_path}]"
    try:
        img = Image.open(image_path).convert("L")
        img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
        psms = [11, 6, 7, 4, 3, 12, 13]
        best_text, best_score, best_psm = "", -1, None
        for psm in psms:
            cfg = f"--oem 1 --psm {psm} -c user_defined_dpi=300"
            text = _ocr_once(img, cfg)
            score = sum(ch.isalnum() for ch in text)
            if score > best_score:
                best_text, best_score, best_psm = text, score, psm
        return best_text
    except Exception as e:
        return f"[fallback_ocr error: {e}]"

# ---------- Utility ----------
def extract_company_offline(text: str) -> str:
    import re
    m = re.search(r"(?i)company[:\.\s]+([A-Za-z0-9 .,&'-]+)", text or "")
    return m.group(1).strip() if m else "[No LLM] Could not extract company name."

def build_fallback_prompt(question: str, ocr_text: str) -> str:
    return (
        f"Given the following OCR text from a document, answer the question directly.\n\n"
        f"OCR TEXT:\n{ocr_text}\n\nQUESTION: {question}\n\nAnswer:"
    )

# ---------- Pipelines ----------
def run_manual_pipeline():
    primary = abs_path("examples/invoice_clean_acme_image.png")
    demo = ensure_demo_invoice()
    user_query = "What is the company name on the invoice?"
    for chosen in (primary, demo):
        if not chosen.exists():
            continue
        content = route_document(str(chosen)) or fallback_ocr_best(chosen)
        if "company" in content.lower():
            response = extract_company_offline(content)
        else:
            response = "[No company info found]"
        print(f"✅ Manual Pipeline → {response}")
        break

def run_langgraph_pipeline():
    primary = abs_path("examples/invoice_clean_acme_image.png")
    demo = ensure_demo_invoice()
    workflow = build_rag_workflow()
    for chosen in (primary, demo):
        if not chosen.exists():
            continue
        state = {"doc_path": str(chosen), "query": "What is the company name on the invoice?"}
        output = workflow.invoke(state) if (groq_api_key or openai_api_key) else {}
        answer = (output or {}).get("answer", "")
        if not answer or "I don't know" in answer:
            text = fallback_ocr_best(chosen)
            answer = extract_company_offline(text)
        print(f"✅ LangGraph Pipeline → {answer}")
        break

def run_agentic_pipeline():
    primary = abs_path("examples/invoice_clean_acme_image.png")
    demo = ensure_demo_invoice()
    for chosen in (primary, demo):
        if not chosen.exists():
            continue

        answer = None

        if openai_api_key or groq_api_key:
            result = agentic_pipeline(
                str(chosen),
                "What is the company name on the invoice?"
            )
            answer = (result or {}).get("answer", "").strip()
        else:
            answer = ""

        # Always fallback if LLM fails or says "I don't know"
        if not answer or "i don't know" in answer.lower():
            text = fallback_ocr_best(chosen)
            answer = extract_company_offline(text)

        print(f"✅ Agentic Pipeline → {answer}")
        break

# ---------- Entry ----------
def main():
    print("✅ Environment loaded. Beginning pipelines...\n")
    run_manual_pipeline()
    run_langgraph_pipeline()
    run_agentic_pipeline()

if __name__ == "__main__":
    main()
