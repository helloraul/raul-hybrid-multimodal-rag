from agents.rag_orchestrator import generate_answer_from_context
from loaders.ocr_router import route_to_loader

def main():
    file_path = "sample_docs/sample1.pdf"
    user_query = "Summarize the key points."

    result = route_to_loader(file_path)
    print("🔍 OCR + Routing Output:", result)

    response = generate_answer_from_context(user_query, result["content"])
    print("💬 Final Response:\n", response)

if __name__ == "__main__":
    main()
