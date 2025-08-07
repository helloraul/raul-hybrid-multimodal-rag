#  Hybrid Multimodal RAG Pipeline

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline that integrates:

- OCR for extracting text from images
- Document routing and chunking
- Hybrid search (keyword + vector)
- Answer generation via Groq (LLaMA 3) with OpenAI fallback

---

##  Architecture

```text
[ Document ]
     ↓
[ OCR Agent (if needed) ]
     ↓
[ Document Router ]
     ↓
[ Chunking ]
     ↓
[ Hybrid Retriever (Keyword + Vector) ]
     ↓
[ LLM (Groq / OpenAI) ]
     ↓
[ Final Answer ]
```

![Pipeline Diagram](A_flowchart_diagram_in_the_image_illustrates_a_hyb.png)

---

##  Setup

### 1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/your-username/raul-hybrid-multimodal-rag.git
cd raul-hybrid-multimodal-rag
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional fallback
```

---

##  Testing

Run all tests with:

```bash
./test_run.sh
```

Or manually:

```bash
python -m pytest -v
```

---

##  File Structure

```
src/
  ├── agents/
  │   └── rag_orchestrator.py
  ├── ocr/
  │   └── ocr_agent.py
  ├── router/
  │   └── doc_router.py
  ├── retrieval/
  │   └── hybrid_search.py

tests/
  ├── test_rag_orchestrator.py
  └── test_ocr_agent.py

.env.example
requirements.txt
test_run.sh
README.md
```

---

##  Usage Example

```python
from agents.rag_orchestrator import generate_answer_from_context

answer = generate_answer_from_context(
    "What is OCR?",
    "This document includes scanned image text using Tesseract."
)

print(answer)
```

---

##  `test_run.sh` – Automated Test Runner

```bash
#!/bin/bash

echo " Running full test suite..."

source venv/bin/activate || source venv/Scripts/activate

pytest -v tests/

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo " All tests passed!"
else
    echo " Some tests failed. Please check the logs above."
fi

exit $exit_code
```

Make it executable:

```bash
chmod +x test_run.sh
```

---

##  `.env.example`

```env
# Copy this to `.env` and fill in real keys

GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

##  Status

- All major components tested
- Handles OCR + routing + hybrid search + LLM fallback
- Includes mocks, real inference test, and error handling
- Diagram and examples included
- Future: caching, UI wrapper, streaming answers