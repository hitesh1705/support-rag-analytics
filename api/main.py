"""FastAPI backend for the support RAG analytics tool."""

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from rag.query import answer as rag_answer

PROCESSED_PATH = "data/processed/tickets"

app = FastAPI(title="Support RAG Analytics API")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Answer a natural-language question grounded in support ticket data."""
    return {"answer": rag_answer(request.question)}


@app.get("/trends")
def trends() -> dict:
    """Return ticket volume over time."""
    df = pd.read_parquet(PROCESSED_PATH)
    counts = df.groupby("date").size().reset_index(name="ticket_count")
    return {"trends": counts.to_dict("records")}


@app.get("/sentiment-summary")
def sentiment_summary() -> dict:
    """Return average sentiment by date."""
    df = pd.read_parquet(PROCESSED_PATH)
    summary = df.groupby("date")["sentiment"].mean().reset_index()
    return {"sentiment_summary": summary.to_dict("records")}
