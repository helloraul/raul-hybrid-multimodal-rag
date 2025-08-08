from langgraph.graph import StateGraph, END
from src.router.doc_router import route_document
from src.retrieval.hybrid_search import retrieve_relevant_chunks
from src.agents.rag_orchestrator import generate_answer_from_context

# Define the graph state
class RAGState(dict): pass

def load_document_node(state: RAGState) -> RAGState:
    print("STATE RECEIVED:", state)
    doc_path = state["doc_path"]
    state["raw_text"] = route_document(doc_path)
    return state

def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    doc = state["raw_text"]
    state["chunks"] = retrieve_relevant_chunks(query, doc)
    return state

def llm_answer_node(state: RAGState) -> RAGState:
    query = state["query"]
    doc = "\n".join(state["chunks"])
    state["answer"] = generate_answer_from_context(query, doc)
    return state



def load_document_node(state: dict) -> dict:
    print("📥 STATE RECEIVED:", state)

    doc_path = state.get("doc_path")
    query = state.get("query")

    if not doc_path or not query:
        raise ValueError("Missing 'doc_path' or 'query' in state")

    content = route_document(doc_path)
    return {"query": query, "context": content}


def generate_node(state: dict) -> dict:
    print("🧠 Generating with context:", state)
    answer = generate_answer_from_context(state["query"], state["context"])
    return {"answer": answer}


def build_rag_workflow():
    workflow = StateGraph(dict)  # ✅ use dict to define input schema

    workflow.add_node("load_document", load_document_node)
    workflow.add_node("generate_answer", generate_node)

    workflow.set_entry_point("load_document")
    workflow.add_edge("load_document", "generate_answer")
    workflow.set_finish_point("generate_answer")

    return workflow.compile()
