"""RAG query engine: retrieves relevant tickets and asks Groq (free, hosted Llama) for a grounded answer."""

import os

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 8

_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(COLLECTION_NAME)
_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def retrieve(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve the top-k most relevant tickets for a question."""
    query_embedding = _model.encode([question]).tolist()
    return _collection.query(query_embeddings=query_embedding, n_results=top_k)


def answer(question: str) -> str:
    """Retrieve relevant tickets and ask Groq to answer grounded in them."""
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

    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    q = "What are customers most frequently complaining about?"
    print(answer(q))
