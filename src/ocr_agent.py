def ocr_and_route(document_path):
    # Placeholder for actual OCR and routing logic
    if "invoice" in document_path.lower():
        return "finance"
    elif "resume" in document_path.lower():
        return "hr"
    else:
        return "general"
