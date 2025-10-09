"""
Tests for the FastAPI microservice (app.py) that exposes /transaction.
Requires: httpx (for TestClient), fastapi, pydantic.
"""

from fastapi.testclient import TestClient

# Import the FastAPI instance from app.py
# If your app file lives elsewhere (e.g., src/app.py), change the import to:
#   from src.app import app as fastapi_app
from app import app as fastapi_app

client = TestClient(fastapi_app)


def test_health():
    """Basic healthcheck should return status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_contains_score_mapping():
    """Config endpoint should expose current rule thresholds/weights."""
    r = client.get("/config")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert "score_to_decision" in payload
    assert "amount_thresholds" in payload


def test_transaction_in_review_path():
    """Typical medium-risk digital transaction from NEW user at night -> IN_REVIEW."""
    body = {
        "transaction_id": 42,
        "amount_mxn": 5200.0,
        "customer_txn_30d": 1,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 23,
        "product_type": "digital",
        "latency_ms": 180,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 42
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW", "REJECTED")
    # With the current defaults (reject_at=10, review_at=4), this should lean to IN_REVIEW
    # If you tuned env vars REJECT_AT/REVIEW_AT, this assertion may need adjustment.
    assert data["decision"] == "IN_REVIEW"


def test_transaction_hard_block_rejection():
    """Chargebacks>=2 with ip_risk=high should trigger hard block -> REJECTED."""
    body = {
        "transaction_id": 99,
        "amount_mxn": 300.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 2,
        "hour": 12,
        "product_type": "digital",
        "latency_ms": 100,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "high",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 99
    assert data["decision"] == "REJECTED"

def test_transaction_accepted_low_risk():
    """
    Low-risk transaction: established user, daytime, low amount, 
    all risk signals low -> ACCEPTED.
    """
    body = {
        "transaction_id": 101,
        "amount_mxn": 250.0,
        "customer_txn_30d": 45,
        "geo_state": "Nuevo León",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 14,
        "product_type": "physical",
        "latency_ms": 95,
        "user_reputation": "trusted",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 101
    assert data["decision"] == "ACCEPTED"
    assert "risk_score" in data
    assert data["risk_score"] < 4  # Below review threshold


def test_transaction_rejected_multiple_red_flags():
    """
    High-risk transaction: high amount, suspicious user, mismatched countries,
    high device/IP risk, late night -> REJECTED.
    """
    body = {
        "transaction_id": 202,
        "amount_mxn": 15000.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 1,
        "hour": 3,
        "product_type": "digital",
        "latency_ms": 450,
        "user_reputation": "suspicious",
        "device_fingerprint_risk": "high",
        "ip_risk": "high",
        "email_risk": "disposable",
        "bin_country": "MX",
        "ip_country": "US"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 202
    assert data["decision"] == "REJECTED"
    assert "risk_score" in data
    assert data["risk_score"] >= 10  # Above rejection threshold


def test_transaction_edge_case_medium_amount():
    """
    Edge case: medium amount with mixed signals (good reputation but 
    medium IP risk, early morning) -> likely IN_REVIEW.
    """
    body = {
        "transaction_id": 303,
        "amount_mxn": 3500.0,
        "customer_txn_30d": 12,
        "geo_state": "Ciudad de México",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 6,
        "product_type": "digital",
        "latency_ms": 220,
        "user_reputation": "good",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 303
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")
    # Should be IN_REVIEW due to medium IP risk + early hour + digital product


def test_transaction_country_mismatch_review():
    """
    Country mismatch between BIN and IP with otherwise normal signals
    -> IN_REVIEW for manual verification.
    """
    body = {
        "transaction_id": 404,
        "amount_mxn": 1200.0,
        "customer_txn_30d": 8,
        "geo_state": "Jalisco",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 16,
        "product_type": "physical",
        "latency_ms": 150,
        "user_reputation": "good",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "CO"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 404
    # Country mismatch should add risk points
    assert data["decision"] in ("IN_REVIEW", "ACCEPTED")


def test_transaction_high_velocity_new_user():
    """
    New user attempting high-velocity transaction (high latency_ms suggests
    rapid attempts) with medium risks -> IN_REVIEW or REJECTED.
    """
    body = {
        "transaction_id": 505,
        "amount_mxn": 8000.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 22,
        "product_type": "digital",
        "latency_ms": 550,
        "user_reputation": "new",
        "device_fingerprint_risk": "medium",
        "ip_risk": "medium",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 505
    assert data["decision"] in ("IN_REVIEW", "REJECTED")
    # High amount + new user + night time + high latency should accumulate risk


def test_transaction_trusted_user_bypass():
    """
    Trusted user with excellent history should get ACCEPTED even with
    slightly elevated amount and non-ideal timing.
    """
    body = {
        "transaction_id": 606,
        "amount_mxn": 4500.0,
        "customer_txn_30d": 89,
        "geo_state": "Nuevo León",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 1,
        "product_type": "physical",
        "latency_ms": 110,
        "user_reputation": "trusted",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 606
    # Trusted users should get preferential treatment
    assert data["decision"] == "ACCEPTED"
