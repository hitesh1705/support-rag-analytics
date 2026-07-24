# Project instructions for Claude Code

## What this project is

An end-to-end RAG analytics tool over customer support tickets. Spark ETL -> embeddings -> vector search -> Claude-generated answers -> FastAPI -> Streamlit dashboard. Full architecture is in README.md.

## Build priorities (in order)

1. `etl/spark_clean.py` - PySpark job that reads raw ticket CSV/JSON from `data/raw/`, cleans text (strip URLs, @handles, emails, phone numbers), computes a basic sentiment score per ticket, and writes partitioned Parquet to `data/processed/`.
2. `rag/embed.py` - loads processed Parquet, generates embeddings with `sentence-transformers` (model: `all-MiniLM-L6-v2`), stores them in a local Chroma collection.
3. `rag/query.py` - given a natural-language question, embeds it, retrieves top-k similar tickets from Chroma, builds a prompt with those tickets as context, calls the Claude API, returns a grounded answer with ticket IDs cited.
4. `api/main.py` - FastAPI app exposing:
   - `POST /ask` - body: `{"question": str}`, returns the RAG answer
   - `GET /trends` - returns ticket volume over time (from processed data)
   - `GET /sentiment-summary` - returns sentiment breakdown by category/date
5. `frontend/app.py` - Streamlit app with a chat box hitting `/ask` and a line/bar chart hitting `/trends` and `/sentiment-summary`.

## Constraints

- Keep the dataset sample small (10k-50k rows) so everything runs fast locally without EMR or a Spark cluster.
- No em dashes or long dashes in any code comments, docstrings, or generated docs.
- Don't fine-tune any models. Use pretrained sentiment (e.g. `vaderSentiment` or a HuggingFace pipeline) and Claude for any zero-shot categorization needed.
- Read the Anthropic API key from the `ANTHROPIC_API_KEY` environment variable. Never hardcode it.
- Keep dependencies in `requirements.txt` current as you add libraries.
- Write short docstrings on functions, not verbose comments.

## Style

- Python, type hints where reasonable.
- Prefer small, testable functions over large scripts.
- Add a basic test in `tests/` for the ETL cleaning function and the RAG retrieval function.
