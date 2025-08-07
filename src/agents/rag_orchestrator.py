import os
import requests
from retrieval.hybrid_search import retrieve_relevant_chunks
from langchain.schema import HumanMessage

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_openai_llm():
    """
    Lazily initialize OpenAI fallback LLM to avoid import-time errors.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def generate_answer_from_context(query: str, document: str, fallback_llm=None) -> str:
    """
    Retrieve relevant chunks and use Groq or fallback LLM to generate an answer.
    """
    context_chunks = retrieve_relevant_chunks(query, document)
    context = "\n".join(context_chunks)

    prompt = f"""
Use the following context to answer the question. If the context is irrelevant or insufficient, say you don't know.

Context:
{context}

Question: {query}
"""

    # 🔹 Step 1: Try Groq
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # 🔸 Step 2: Fall back to OpenAI or injected LLM
    except Exception as e:
        print(f"⚠️ Groq failed: {e}, falling back to OpenAI...")
        try:
            llm = fallback_llm or get_openai_llm()
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as openai_error:
            return f"❌ OpenAI fallback also failed: {openai_error}"
