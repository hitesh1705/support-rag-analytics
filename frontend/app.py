"""Streamlit dashboard: chat over support tickets plus trend charts."""

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Support RAG Analytics", layout="wide")
st.title("Support RAG Analytics")

tab_chat, tab_trends = st.tabs(["Ask", "Trends"])

with tab_chat:
    question = st.text_input("Ask a question about your support tickets")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/ask", json={"question": question})
            st.write(response.json().get("answer", "No answer returned."))

with tab_trends:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ticket volume over time")
        trend_data = requests.get(f"{API_URL}/trends").json()["trends"]
        df_trends = pd.DataFrame(trend_data)
        if not df_trends.empty:
            st.line_chart(df_trends.set_index("date"))

    with col2:
        st.subheader("Sentiment over time")
        sentiment_data = requests.get(f"{API_URL}/sentiment-summary").json()["sentiment_summary"]
        df_sentiment = pd.DataFrame(sentiment_data)
        if not df_sentiment.empty:
            st.line_chart(df_sentiment.set_index("date"))
