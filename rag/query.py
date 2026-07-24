"""RAG query engine: retrieves relevant tickets and asks Claude for a grounded answer."""

import os

import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "support_tickets"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 8

_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(COLLECTION_NAME)
_anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def retrieve(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve the top-k most relevant tickets for a question."""
    query_embedding = _model.encode([question]).tolist()
    return _collection.query(query_embeddings=query_embedding, n_results=top_k)


def answer(question: str) -> str:
    """Retrieve relevant tickets and ask Claude to answer grounded in them."""
    results = retrieve(question)
    documents = results["documents"][0]
    ids = results["ids"][0]

    context_blocks = [
        f"[Ticket {tid}]: {doc}" for tid, doc in zip(ids, documents)
    ]
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a support analytics assistant. Answer the question using only "
        "the ticket excerpts below. Cite ticket IDs for any claim you make.\n\n"
        f"Tickets:\n{context}\n\nQuestion: {question}"
    )

    response = _anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(
        block.text for block in response.content if block.type == "text"
    )


if __name__ == "__main__":
    q = "What are customers most frequently complaining about?"
    print(answer(q))
