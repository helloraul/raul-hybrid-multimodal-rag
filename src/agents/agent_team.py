class OcrAgent:
    def run(self, file_path: str) -> str:
        # Call actual OCR logic here
        from src.ocr.ocr_agent import extract_text_from_image
        return extract_text_from_image(file_path)

class SummarizerAgent:
    def run(self, content: str, query: str) -> str:
        from src.agents.rag_orchestrator import generate_answer_from_context
        return generate_answer_from_context(query, content)

class ClassifierAgent:
    def run(self, content: str) -> str:
        # Dummy classification (simulate a specialized agent)
        if "invoice" in content.lower():
            return "Finance"
        elif "meeting" in content.lower():
            return "Notes"
        else:
            return "General"
