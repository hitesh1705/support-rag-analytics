# Support RAG Analytics

An end-to-end pipeline that turns raw customer support tickets into a searchable, queryable knowledge base. Spark handles cleaning and feature extraction, embeddings power semantic search, and a free hosted LLM generates grounded answers to natural-language questions like "what are the top complaints in October 2017?"

## Screenshots

**Ask tab, natural-language Q&A over ticket history:**
![Ask tab](/Users/hiteshyarlagadda/support-rag-analytics/screenshots/ask-tab-screenshot)

**Trends tab, ticket volume and sentiment over time:**
![Trends tab](/Users/hiteshyarlagadda/support-rag-analytics/screenshots/trends-tab-screenshot)

## Architecture

```
Raw tickets (Twitter customer support dataset)
        |
        v
   Local CSV (data/raw/tickets.csv)
        |
        v
  PySpark ETL job --> clean text, strip PII, VADER sentiment score, partition by date
        |
        v
  Partitioned Parquet (data/processed/tickets)
        |
        v
  Embedding pipeline (sentence-transformers, all-MiniLM-L6-v2)
        |
        v
  Chroma vector store (local, persistent)
        |
        v
  RAG query engine --> retrieves top-k tickets, calls Groq API (Llama 3.3 70B) for grounded answer
        |
        v
  FastAPI backend (/ask, /trends, /sentiment-summary)
        |
        v
  Streamlit dashboard (chat + trend charts)
```

## Why this project exists

Most portfolio "RAG chatbot" repos skip the data engineering step entirely. This one doesn't: the ETL layer is a real Spark job doing real cleaning, PII stripping, and sentiment scoring, not just a CSV read. The goal is to demonstrate the full path from messy raw data to a working, LLM-powered analytics tool.

## Tech stack

- **ETL**: Apache Spark (PySpark), run locally
- **Data**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (Kaggle), 20k-row sample
- **Sentiment**: VADER (rule-based sentiment analysis)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`), local, free
- **Vector store**: Chroma (local, persistent)
- **LLM / RAG generation**: [Groq API](https://console.groq.com) running Llama 3.3 70B, free tier, no local GPU needed
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Built with**: Claude Code (see `CLAUDE.md` for how this project was scaffolded and iterated on)

## Setup

```bash
# 1. Clone
git clone https://github.com/hitesh1705/support-rag-analytics.git
cd support-rag-analytics

# 2. Create and activate a virtual environment (Python 3.11 recommended for PySpark compatibility)
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Kaggle API access (for dataset download)
# Get your token from kaggle.com/settings, then:
mkdir -p ~/.kaggle
# create ~/.kaggle/kaggle.json with {"username": "...", "key": "..."}
chmod 600 ~/.kaggle/kaggle.json

# 5. Download and sample the dataset
python scripts/download_dataset.py

# 6. Run the Spark ETL job
python etl/spark_clean.py

# 7. Build embeddings + vector store
python rag/embed.py

# 8. Get a free Groq API key at console.groq.com, no credit card required
export GROQ_API_KEY=your_key_here

# 9. Start the API
uvicorn api.main:app --reload

# 10. Start the dashboard (separate terminal, with venv activated)
streamlit run frontend/app.py
```

## Design decisions worth knowing about

- **Local Spark instead of EMR**: this project runs Spark locally/via Docker rather than on a distributed cluster, since the dataset is a 20k-row sample sized for fast iteration. The ETL logic (cleaning, PII stripping, partitioning, sentiment scoring) is written the same way it would be on a cluster; scaling to EMR Serverless would mean pointing the job at S3 paths instead of local paths.
- **Groq instead of a paid LLM API**: Groq's free tier hosts open-source models (Llama 3.3 70B) with no cost and no rate limits that matter for a project this size, which keeps the whole pipeline genuinely free to run and reproduce.
- **VADER instead of a fine-tuned sentiment model**: rule-based sentiment is fast, deterministic, and good enough for trend analysis at this scale, without needing to fine-tune anything.

## Roadmap

- Distributed processing on AWS EMR Serverless
- Airflow orchestration for scheduled re-ingestion
- OpenSearch Serverless for vector storage at scale
- Deployment on AWS EC2 (free tier)

## License

MIT
