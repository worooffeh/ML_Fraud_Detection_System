from __future__ import annotations

import os

import requests
import streamlit as st


def render_fraud_app() -> None:
    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    st.set_page_config(page_title="Nova Pay Fraud Scoring", layout="centered")
    st.title("Nova Pay Fraud Scoring")
    st.caption("Submit a transaction to the production scoring API.")

    with st.form("fraud_score_form"):
        left, right = st.columns(2)
        with left:
            velocity_1h = st.number_input(
                "Transactions in the last hour", min_value=0.0, value=1.0, step=1.0
            )
            velocity_24h = st.number_input(
                "Transactions in the last 24 hours", min_value=0.0, value=3.0, step=1.0
            )
            ip_risk = st.number_input(
                "IP risk score", min_value=0.0, max_value=1.0, value=0.1, step=0.01
            )
        with right:
            device_trust = st.number_input(
                "Device trust score", min_value=0.0, max_value=1.0, value=0.9, step=0.01
            )
            location_mismatch = st.selectbox(
                "Country/location mismatch",
                options=[0, 1],
                format_func=lambda value: "Yes" if value else "No",
            )
            amount_usd = st.number_input(
                "Amount (USD)", min_value=0.0, value=50.0, step=1.0
            )
        submitted = st.form_submit_button(
            "Score transaction", type="primary", use_container_width=True
        )

    if not submitted:
        return

    payload = {
        "txn_velocity_1h": velocity_1h,
        "txn_velocity_24h": velocity_24h,
        "ip_risk_score": ip_risk,
        "device_trust_score": device_trust,
        "country_location_mismatch": location_mismatch,
        "amount_usd": amount_usd,
    }
    try:
        response = requests.post(f"{api_base_url}/score", json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Scoring API unavailable: {exc}")
        return

    result = response.json()
    st.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    st.metric("Decision", "Fraud" if result["is_fraud"] else "Legitimate")
    st.caption(f"Operational threshold: {result['threshold']:.2f}")


render_fraud_app()