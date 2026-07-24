"""PySpark ETL job: cleans raw support ticket data and writes partitioned Parquet.

Reads from data/raw/, strips PII (emails, phone numbers, @handles), computes a
basic sentiment score, and writes partitioned output to data/processed/.
"""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

RAW_PATH = "data/raw/tickets.csv"
PROCESSED_PATH = "data/processed/tickets"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Strip PII, URLs, and handles from a ticket's text."""
    if not text:
        return ""
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = HANDLE_RE.sub("[handle]", text)
    return text.strip()


_vader = SentimentIntensityAnalyzer()


def sentiment_score(text: str) -> float:
    """Compound sentiment score from VADER, ranges from -1 (negative) to 1 (positive)."""
    if not text:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])


def run() -> None:
    spark = SparkSession.builder.appName("support-ticket-etl").getOrCreate()

    clean_udf = udf(clean_text, StringType())
    sentiment_udf = udf(sentiment_score, DoubleType())

    df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(RAW_PATH)
)

    cleaned = (
        df.withColumn("clean_text", clean_udf(col("text")))
        .withColumn("sentiment", sentiment_udf(col("clean_text")))
        .filter(col("clean_text") != "")
    )

    cleaned.write.mode("overwrite").partitionBy("date").parquet(PROCESSED_PATH)

    spark.stop()


if __name__ == "__main__":
    run()
