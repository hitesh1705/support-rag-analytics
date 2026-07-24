"""Downloads and samples the Customer Support on Twitter dataset.

Requires a Kaggle account and API token (~/.kaggle/kaggle.json).
See https://www.kaggle.com/docs/api for setup instructions.

Usage:
    python scripts/download_dataset.py
"""

import subprocess
from pathlib import Path

import pandas as pd

DATASET = "thoughtvector/customer-support-on-twitter"
RAW_DIR = Path("data/raw")
SAMPLE_SIZE = 20000


def download() -> None:
    """Download the dataset via the Kaggle CLI."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )


def find_source_csv() -> Path:
    """Locate the real twcs.csv, which Kaggle unzips into a twcs/ subfolder.

    This dataset also ships a small sample.csv at the top level, so we search
    recursively and pick the largest CSV found rather than the first match.
    """
    candidates = list(RAW_DIR.rglob("*.csv"))
    candidates = [c for c in candidates if c.name != "tickets.csv"]
    if not candidates:
        raise FileNotFoundError("No CSV found after download. Check the Kaggle download step.")
    return max(candidates, key=lambda p: p.stat().st_size)


def sample() -> None:
    """Sample a manageable subset and write it as tickets.csv."""
    source_file = find_source_csv()
    print(f"Reading source file: {source_file}")

    df = pd.read_csv(source_file)
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    # Normalize expected columns for the ETL job: text, date
    if "created_at" in df.columns:
        df["date"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True, format="mixed"
        ).dt.date.astype(str)
    else:
        df["date"] = "unknown"

    if "text" not in df.columns:
        raise KeyError("Expected a 'text' column in the source dataset.")

    df[["text", "date"]].to_csv(RAW_DIR / "tickets.csv", index=False)
    print(f"Wrote {len(df)} sampled tickets to {RAW_DIR / 'tickets.csv'}")

if __name__ == "__main__":
    download()
    sample()
