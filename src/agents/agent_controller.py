from src.agents.agent_team import OcrAgent, SummarizerAgent, ClassifierAgent

def agentic_pipeline(doc_path: str, query: str):
    ocr_agent = OcrAgent()
    classifier = ClassifierAgent()
    summarizer = SummarizerAgent()

    # Step 1: Extract content
    content = ocr_agent.run(doc_path)

    # Step 2: Classify document
    category = classifier.run(content)
    print(f"📂 Document classified as: {category}")

    # Step 3: Summarize or Answer
    response = summarizer.run(content, query)

    return {
        "category": category,
        "answer": response
    }
