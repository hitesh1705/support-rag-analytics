"""Tests for the ETL cleaning function."""

from etl.spark_clean import clean_text, sentiment_score


def test_clean_text_strips_email():
    assert "[email]" in clean_text("contact me at foo@bar.com")


def test_clean_text_strips_handle():
    assert "[handle]" in clean_text("hey @supportteam help me")


def test_clean_text_strips_url():
    assert "http" not in clean_text("check https://example.com for info")


def test_sentiment_score_positive():
    assert sentiment_score("thanks, this was awesome and resolved quickly") > 0


def test_sentiment_score_negative():
    assert sentiment_score("this is the worst, terrible and broken") < 0
