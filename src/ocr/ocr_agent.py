import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import os

# If tesseract is not on PATH, specify the full path here
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f" OCR failed for {image_path}: {e}")
        return "OCR extraction failed."

# Optional routing logic based on text content
def ocr_and_route(document_path: str) -> str:
    text = extract_text_from_image(document_path)
    if "invoice" in text.lower():
        return "finance"
    elif "resume" in text.lower():
        return "hr"
    else:
        return "general"

def perform_ocr(image_path: str) -> str:
    try:
        print(f"📥 Performing OCR on: {image_path}")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text else "⚠️ OCR produced no readable text."
    except FileNotFoundError as e:
        return f" OCR failed for {image_path}: {str(e)}"
    except Exception as e:
        return f" OCR error: {str(e)}"