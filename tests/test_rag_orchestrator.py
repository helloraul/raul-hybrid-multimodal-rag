import sys
import os
import pytest
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.rag_orchestrator import generate_answer_from_context

@pytest.fixture
def mock_context(monkeypatch):
    # Mock retrieval to return dummy chunks
    monkeypatch.setattr(
        "agents.rag_orchestrator.retrieve_relevant_chunks",
        lambda query, doc: ["This is a mocked chunk about pandas.", "Pandas are black and white animals."]
    )

@pytest.fixture
def mock_llm():
    # Mock fallback LLM class with invoke method
    class MockLLM:
        def invoke(self, messages):
            return type('obj', (object,), {"content": "This is a mock fallback answer."})
    return MockLLM()

def test_generate_answer_from_context_with_fallback(mock_context, mock_llm):
    result = generate_answer_from_context("What do pandas look like?", "Some doc content", fallback_llm=mock_llm)
    assert "mock fallback answer" in result


def test_generate_answer_from_context_with_groq(monkeypatch, mock_context):
    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": "🐼 Pandas are cute, slow-moving mammals found in China."}}
                ]
            }

    monkeypatch.setattr("agents.rag_orchestrator.requests.post", lambda *args, **kwargs: MockResponse())

    result = generate_answer_from_context("Tell me about pandas", "Doc content")
    assert "Pandas" in result or "pandas" in result

import os


def test_generate_answer_with_real_groq(monkeypatch):
    from agents import rag_orchestrator

    # ✅ Load API keys from .env (or system env)
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")

    # ✅ Skip test if key is missing (prevent accidental push with real key)
    if not groq_key:
        pytest.skip("GROQ_API_KEY not set. Skipping real Groq integration test.")

    # ✅ Set environment variable without hardcoding
    os.environ["GROQ_API_KEY"] = groq_key

    # 🧪 Mock the retriever
    monkeypatch.setattr(
        "agents.rag_orchestrator.retrieve_relevant_chunks",
        lambda q, d: ["Pandas are mammals native to China."]
    )

    result = rag_orchestrator.generate_answer_from_context("Where do pandas live?", "irrelevant doc")

    # ✅ Gracefully skip if LLM call fails
    if "❌" in result or "fallback" in result.lower():
        pytest.skip("Skipped because Groq call or fallback failed.")
    else:
        assert "China" in result or "china" in result