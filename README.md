# Support RAG Analytics

An end-to-end pipeline that turns raw customer support tickets into a searchable, queryable knowledge base. Spark handles cleaning and feature extraction, embeddings power semantic search, and Claude generates grounded answers to natural-language questions like "what are customers complaining about with shipping this month?"

## Architecture

```
Raw tickets (CSV/JSON)
        |
        v
   S3 (raw bucket)
        |
        v
  PySpark ETL job  --> clean text, strip PII, sentiment score, partition by date
        |
        v
  S3 (processed bucket, Parquet)
        |
        v
  Embedding pipeline (sentence-transformers)
        |
        v
  Vector store (Chroma / FAISS)
        |
        v
  RAG query engine  --> retrieves top-k tickets, calls Claude API for grounded answer
        |
        v
  FastAPI backend  (/ask, /trends, /sentiment-summary)
        |
        v
  Streamlit dashboard (chat + trend charts)
```

## Why this project exists

Most portfolio "RAG chatbot" repos skip the data engineering step entirely. This one doesn't: the ETL layer is a real Spark job doing real cleaning and feature extraction, not just a CSV read. The goal is to demonstrate the full path from messy raw data to a production-shaped, LLM-powered analytics tool.

## Tech stack

- **ETL**: Apache Spark (PySpark), Dockerized
- **Storage**: AWS S3 (free tier)
- **Embeddings**: sentence-transformers (local, free)
- **Vector store**: Chroma
- **LLM / RAG**: Claude API (Anthropic)
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Deployment**: AWS EC2 (free tier, t3.micro)
- **Built with**: Claude Code (see `CLAUDE.md` for how this project was scaffolded and iterated on)

## Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd support-rag-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 4. Download sample dataset
python scripts/download_dataset.py

# 5. Run the Spark ETL job
python etl/spark_clean.py

# 6. Build embeddings + vector store
python rag/embed.py

# 7. Start the API
uvicorn api.main:app --reload

# 8. Start the dashboard (separate terminal)
streamlit run frontend/app.py
```

## Dataset

Uses a sample of the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. `scripts/download_dataset.py` documents how to fetch it (requires a free Kaggle account/API key).

## Deployment (AWS free tier)

- S3 for raw and processed data storage
- EC2 t3.micro (free tier, 750 hrs/month for 12 months) running the FastAPI + Streamlit app via Docker Compose
- See `infra/` for setup notes (added as the project develops)

## Project status

Built as a rapid, focused portfolio project. Roadmap for future iterations: distributed processing on EMR Serverless, Airflow orchestration, OpenSearch Serverless for vector storage at scale.

## License

MIT
