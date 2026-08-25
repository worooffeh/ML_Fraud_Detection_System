"""Production smoke test for the Nova Pay API and Streamlit UI."""

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
UI_BASE_URL = os.getenv("UI_BASE_URL", "http://127.0.0.1:8501").rstrip("/")

SAMPLE_TRANSACTION = {
    "txn_velocity_1h": 2.0,
    "txn_velocity_24h": 8.0,
    "ip_risk_score": 0.1,
    "device_trust_score": 0.9,
    "country_location_mismatch": 0,
    "amount_usd": 42.50,
}


def request_json(url, method="GET", payload=None):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=10) as response:
        raw_body = response.read().decode("utf-8")
        return response.status, json.loads(raw_body) if raw_body else None


def request_text(url):
    with urlopen(url, timeout=10) as response:
        return response.status, response.read().decode("utf-8").strip()


def check_api():
    status, response = request_json(f"{API_BASE_URL}/health")
    if status != 200 or response != {"status": "ok"}:
        raise AssertionError(f"API health check failed: {status} {response}")
    print(f"PASS API health: {response}")

    status, response = request_json(f"{API_BASE_URL}/ready")
    if status != 200 or response.get("status") != "ready":
        raise AssertionError(f"API readiness check failed: {status} {response}")
    print(f"PASS API ready: {response}")

    status, response = request_json(
        f"{API_BASE_URL}/score", method="POST", payload=SAMPLE_TRANSACTION
    )
    probability = response.get("fraud_probability")
    if (
        status != 200
        or not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
        or not isinstance(response.get("is_fraud"), bool)
        or not 0 <= response.get("threshold", -1) <= 1
    ):
        raise AssertionError(f"API scoring check failed: {status} {response}")
    print(f"PASS API score: {response}")


def check_ui():
    status, response = request_text(f"{UI_BASE_URL}/_stcore/health")
    if status != 200 or response != "ok":
        raise AssertionError(f"UI health check failed: {status} {response!r}")
    print(f"PASS UI health: {response}")


def main():
    try:
        check_api()
        check_ui()
    except (AssertionError, HTTPError, URLError, TimeoutError, ValueError) as error:
        print(f"FAIL production smoke test: {error}", file=sys.stderr)
        return 1

    print("Production smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
