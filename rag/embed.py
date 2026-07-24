"""Builds a Chroma vector store from the processed ticket Parquet files."""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

PROCESSED_PATH = "data/processed/tickets"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "support_tickets"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_processed() -> pd.DataFrame:
    """Load processed ticket data from partitioned Parquet."""
    return pd.read_parquet(PROCESSED_PATH)


def build_vector_store() -> None:
    """Embed cleaned ticket text and store in a local Chroma collection."""
    df = load_processed()
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    texts = df["clean_text"].tolist()
    ids = [str(i) for i in df.index]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    metadatas = df[["date", "sentiment"]].astype(str).to_dict("records")

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(texts)} tickets into Chroma collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_vector_store()
