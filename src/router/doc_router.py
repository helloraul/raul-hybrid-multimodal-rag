import os
from ocr.ocr_agent import extract_text_from_image


def route_document(doc_path: str) -> str:
    """
    Intelligent document router that determines if OCR is needed.
    """
    ext = os.path.splitext(doc_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        print("🧠 Routing through OCR Agent...")
        return extract_text_from_image(doc_path)

    elif ext in [".pdf"]:
        # Add smart logic later (e.g., check if it's a scanned PDF)
        print("📄 PDF detected. OCR routing logic not yet implemented.")
        return "PDF OCR not yet supported."

    else:
        print("✅ Text-based document detected.")
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()
