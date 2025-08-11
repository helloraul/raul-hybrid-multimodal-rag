# Hybrid Multimodal RAG Pipeline

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline that integrates:

- OCR for extracting text from images
- Document routing and chunking
- Hybrid search (keyword + vector)
- Answer generation via Groq (LLaMA 3) with OpenAI fallback
- Multiple execution modes: Manual, LangGraph, and Agentic

---

## 📌 Current Status (as of Aug 7, 2025)

### ✅ Implemented & Working
- **OCR + Routing**
  - Primary OCR Agent (`route_document`) with preprocessing
  - Fallback OCR (`fallback_ocr_best`) with multiple PSM modes for accuracy improvement
  - Automatic extraction of company name from invoices
- **Three Pipeline Modes**
  - **Manual Pipeline** – Direct OCR extraction
  - **LangGraph Pipeline** – Orchestrated workflow with OCR + LLM fallback
  - **Agentic Pipeline** – Agent-based reasoning with OCR fallback when LLM confidence is low
- **FastAPI Service**
  - `/healthz` and `/llm-ping` endpoints for health checks
  - `/debug/extract` endpoint to preview document extraction
  - `/eval` endpoint (BM25 + fallback context)
  - `/ask` endpoint (BM25 + TF-IDF + guardrails, with citations)
- **Docker Integration**
  - Application runs in Docker with `docker-compose`
  - Dataset mounted via volume mapping (`./SampleDataSet` → `/app/SampleDataSet`)
- **Error Handling**
  - Graceful handling of missing `.env` or API keys (OpenAI, Grok)
  - Automatic fallback to demo invoice image if file is missing
- **Extended Docker Compose Setup**
  - Add Neo4j, Arize, and any multilingual dependencies as services
- **Neo4j Integration**
  - Export structured entities & relationships
  - Implement `use_neo4j=true` retrieval path in `/ask`

---

### 🚧 Pending / Next Steps
- **Retriever Quality Improvements**
  - Enhance BM25 + vector retrieval accuracy for financial PDFs
  - Increase relevant context passed to LLM for better answers
- **Multilingual OCR & Query Support**
  - Language detection and translation for queries/documents
- **Arize Phoenix Observability**
  - Trace RAG pipeline stages, measure LLM latency, and monitor retrieval quality
- **Evaluation Metrics**
  - Implement Token-F1, answer accuracy, and latency reporting
- 


---

## 🏗 Architecture

```text
[ Document / Image ]
     ↓
[ OCR Agent (primary + fallback) ]
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


⚙ Setup
1. Clone the repo and create a virtual environment:
bash
Copy
Edit
git clone https://github.com/your-username/raul-hybrid-multimodal-rag.git
cd raul-hybrid-multimodal-rag
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Configure environment variables:
bash
Copy
Edit
cp .env.example .env
Edit .env with your API keys:

env
Copy
Edit
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional fallback
▶ How to Run
Run the main script to execute all pipelines sequentially:

bash
Copy
Edit
python src/main.py
This will:

Manual Pipeline → Runs OCR extraction only

LangGraph Pipeline → Orchestrates RAG with OCR + LLM fallback

Agentic Pipeline → Runs agent-based reasoning with OCR fallback

Sample output:

mathematica
Copy
Edit
 Manual Pipeline → Acme Robotics Inc
 LangGraph Pipeline → Acme Robotics Inc.
 Agentic Pipeline → Acme Robotics Inc.
 Note:

If examples/demo_invoice.png is missing, the script will generate a sample invoice automatically.

You can swap in your own test files in the examples/ directory.

🧪 Testing
Run all tests with:

bash
Copy
Edit
./test_run.sh
Or manually:

bash
Copy
Edit
python -m pytest -v
📂 File Structure
css
Copy
Edit
src/
  ├── agents/
  │   ├── agent_controller.py
  │   └── agent_team.py
  ├── graph/
  ├── ocr/
  │   └── ocr_agent.py
  ├── retrieval/
  │   └── hybrid_search.py
  ├── router/
  │   └── doc_router.py
  └── main.py

examples/
sample_docs/
tests/
.env.example
env
Copy
Edit
# Copy this to `.env` and fill in real keys

GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here


curl -s http://localhost:8000/debug/ask-context \
  -H "Content-Type: application/json" \
  --data-binary @req.json | python -m json.tool

Expect: strategy: "probe" and the preview includes “Total net sales”.

curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  --data-binary @req.json | python -m json.tool

Expect: lines like YYYY Q#: $… and citations in parentheses.

Behavior:
If BM25/TF-IDF can’t confidently select pages or the LLM is unavailable, the API falls back to a heuristic parser that scans 10-Q text for “Total net sales” near “Three/Nine Months Ended” and returns quarterly totals with file citations.

Smoke test:

graphql
Copy
Edit
make ctx   # or the curl debug command above
make ask   # should print YYYY Q# totals + citations
Example output:

bash
Copy
Edit
2022 Q3: $82,959 · $81,434 · $304,182 · $282,457
2023 Q1: $117,154 · $123,945
2023 Q2: $94,836 · $97,278 · $211,990 · $221,223 (2022 Q3 AAPL.pdf; 2023 Q1 AAPL.pdf; 2023 Q2 AAPL


## 📹 Demo Video

Click the image below to watch the demo:

[https://youtu.be/YNVVzHwbBiU]

---

### 🔹 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/helloraul/raul-hybrid-multimodal-rag/YOUR_REPO.git
cd raul-hybrid-multimodal-rag

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# 3. Start the API server
uvicorn src.main:app --reload

# 4. Test with example request
curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  --data-binary @req.json | python -m json.tool

