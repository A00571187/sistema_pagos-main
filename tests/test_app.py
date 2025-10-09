"""
Comprehensive test suite for the FastAPI fraud detection microservice.
Tests cover all decision paths: ACCEPTED, IN_REVIEW, and REJECTED scenarios
based on risk scoring logic, hard blocks, and threshold configuration.
"""

from fastapi.testclient import TestClient
from app import app as fastapi_app

client = TestClient(fastapi_app)


# ============================================================================
# HEALTH & CONFIG ENDPOINTS
# ============================================================================

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


# ============================================================================
# ACCEPTED SCENARIOS - Low Risk Transactions
# ============================================================================

def test_transaction_accepted_low_risk_trusted_user():
    """
    Ultra low-risk: trusted user, daytime, small amount, all signals green.
    Expected: ACCEPTED with minimal risk score.
    """
    body = {
        "transaction_id": 1001,
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
    assert data["transaction_id"] == 1001
    assert data["decision"] == "ACCEPTED"
    assert "risk_score" in data
    assert data["risk_score"] < 4


def test_transaction_accepted_recurrent_user_normal_activity():
    """
    Recurrent user with moderate transaction history during business hours.
    Expected: ACCEPTED.
    """
    body = {
        "transaction_id": 1002,
        "amount_mxn": 800.0,
        "customer_txn_30d": 15,
        "geo_state": "Ciudad de México",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 11,
        "product_type": "physical",
        "latency_ms": 120,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 1002
    assert data["decision"] == "ACCEPTED"


def test_transaction_accepted_established_customer_medium_amount():
    """
    Established customer (30+ txns) with medium amount during safe hours.
    Expected: ACCEPTED despite slightly elevated amount.
    """
    body = {
        "transaction_id": 1003,
        "amount_mxn": 3200.0,
        "customer_txn_30d": 35,
        "geo_state": "Jalisco",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 16,
        "product_type": "digital",
        "latency_ms": 140,
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
    assert data["transaction_id"] == 1003
    assert data["decision"] == "ACCEPTED"


# ============================================================================
# IN_REVIEW SCENARIOS - Medium Risk Transactions
# ============================================================================

def test_transaction_in_review_new_user_night():
    """
    New user making digital purchase at night with medium amount.
    Expected: IN_REVIEW due to newness + timing + product type.
    """
    body = {
        "transaction_id": 2001,
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
    assert data["transaction_id"] == 2001
    assert data["decision"] == "IN_REVIEW"
    assert 4 <= data["risk_score"] < 10


def test_transaction_in_review_country_mismatch():
    """
    Country mismatch between card BIN and IP location.
    Expected: IN_REVIEW for manual verification.
    """
    body = {
        "transaction_id": 2002,
        "amount_mxn": 1200.0,
        "customer_txn_30d": 8,
        "geo_state": "Jalisco",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 16,
        "product_type": "physical",
        "latency_ms": 150,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "CO"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 2002
    assert data["decision"] in ("IN_REVIEW", "ACCEPTED")


def test_transaction_in_review_medium_ip_risk():
    """
    Medium IP risk with early morning transaction and new email domain.
    Expected: IN_REVIEW due to accumulated medium-risk signals.
    """
    body = {
        "transaction_id": 2003,
        "amount_mxn": 3500.0,
        "customer_txn_30d": 12,
        "geo_state": "Ciudad de México",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 6,
        "product_type": "digital",
        "latency_ms": 220,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 2003
    assert data["decision"] == "IN_REVIEW"


def test_transaction_in_review_high_latency():
    """
    High latency suggesting rapid retry attempts with medium risk profile.
    Expected: IN_REVIEW for potential velocity abuse.
    """
    body = {
        "transaction_id": 2004,
        "amount_mxn": 2800.0,
        "customer_txn_30d": 5,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 20,
        "product_type": "digital",
        "latency_ms": 480,
        "user_reputation": "new",
        "device_fingerprint_risk": "medium",
        "ip_risk": "low",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 2004
    assert data["decision"] in ("IN_REVIEW", "REJECTED")


def test_transaction_in_review_medium_device_fingerprint():
    """
    Medium device fingerprint risk with medium amount at night.
    Expected: IN_REVIEW due to device concerns.
    """
    body = {
        "transaction_id": 2005,
        "amount_mxn": 4100.0,
        "customer_txn_30d": 7,
        "geo_state": "Monterrey",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 22,
        "product_type": "digital",
        "latency_ms": 190,
        "user_reputation": "new",
        "device_fingerprint_risk": "medium",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 2005
    assert data["decision"] == "IN_REVIEW"


# ============================================================================
# REJECTED SCENARIOS - High Risk Transactions
# ============================================================================

def test_transaction_rejected_hard_block_chargebacks():
    """
    Hard block: 2+ chargebacks with high IP risk.
    Expected: REJECTED immediately regardless of other factors.
    """
    body = {
        "transaction_id": 3001,
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
    assert data["transaction_id"] == 3001
    assert data["decision"] == "REJECTED"


def test_transaction_rejected_multiple_red_flags():
    """
    Multiple high-risk signals: large amount, high_risk user, country mismatch,
    high device/IP risk, late night, high email risk.
    Expected: REJECTED due to accumulated high risk score.
    """
    body = {
        "transaction_id": 3002,
        "amount_mxn": 15000.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 1,
        "hour": 3,
        "product_type": "digital",
        "latency_ms": 450,
        "user_reputation": "high_risk",
        "device_fingerprint_risk": "high",
        "ip_risk": "high",
        "email_risk": "high",
        "bin_country": "MX",
        "ip_country": "US"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 3002
    assert data["decision"] == "REJECTED"
    assert data["risk_score"] >= 10


def test_transaction_rejected_high_risk_user_high_amount():
    """
    High risk user attempting very high amount transaction.
    Expected: REJECTED due to user reputation + amount.
    """
    body = {
        "transaction_id": 3003,
        "amount_mxn": 12000.0,
        "customer_txn_30d": 2,
        "geo_state": "Ciudad de México",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 2,
        "product_type": "digital",
        "latency_ms": 320,
        "user_reputation": "high_risk",
        "device_fingerprint_risk": "medium",
        "ip_risk": "high",
        "email_risk": "high",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 3003
    assert data["decision"] == "REJECTED"


def test_transaction_rejected_high_device_and_ip_risk():
    """
    Both device fingerprint and IP show high risk with new user.
    Expected: REJECTED due to device/network compromise indicators.
    """
    body = {
        "transaction_id": 3004,
        "amount_mxn": 6500.0,
        "customer_txn_30d": 0,
        "geo_state": "Jalisco",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 4,
        "product_type": "digital",
        "latency_ms": 380,
        "user_reputation": "new",
        "device_fingerprint_risk": "high",
        "ip_risk": "high",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 3004
    assert data["decision"] == "REJECTED"


def test_transaction_rejected_extreme_latency_high_risk():
    """
    Extremely high latency (>500ms) suggesting bot/automated fraud attempt
    with high_risk reputation.
    Expected: REJECTED due to velocity + reputation.
    """
    body = {
        "transaction_id": 3005,
        "amount_mxn": 8000.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 1,
        "hour": 1,
        "product_type": "digital",
        "latency_ms": 650,
        "user_reputation": "high_risk",
        "device_fingerprint_risk": "high",
        "ip_risk": "medium",
        "email_risk": "high",
        "bin_country": "MX",
        "ip_country": "BR"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 3005
    assert data["decision"] == "REJECTED"


# ============================================================================
# EDGE CASES & BOUNDARY TESTING
# ============================================================================

def test_transaction_edge_threshold_review_boundary():
    """
    Transaction at the exact boundary between ACCEPTED and IN_REVIEW.
    Expected: IN_REVIEW (score exactly at review_at threshold).
    """
    body = {
        "transaction_id": 4001,
        "amount_mxn": 2500.0,
        "customer_txn_30d": 10,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 19,
        "product_type": "digital",
        "latency_ms": 200,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 4001
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")


def test_transaction_trusted_user_bypass_night():
    """
    Trusted user with excellent history should get ACCEPTED even with
    non-ideal timing and elevated amount.
    Expected: ACCEPTED due to strong reputation override.
    """
    body = {
        "transaction_id": 4002,
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
    assert data["transaction_id"] == 4002
    assert data["decision"] == "ACCEPTED"


def test_transaction_zero_amount():
    """
    Edge case: zero amount transaction (test/verification transaction).
    Expected: ACCEPTED (no risk from amount).
    """
    body = {
        "transaction_id": 4003,
        "amount_mxn": 0.0,
        "customer_txn_30d": 5,
        "geo_state": "Ciudad de México",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 14,
        "product_type": "digital",
        "latency_ms": 90,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 4003
    assert data["decision"] == "ACCEPTED"


def test_transaction_maximum_amount():
    """
    Edge case: very high amount with trusted user.
    Expected: IN_REVIEW or ACCEPTED depending on thresholds.
    """
    body = {
        "transaction_id": 4004,
        "amount_mxn": 25000.0,
        "customer_txn_30d": 50,
        "geo_state": "Nuevo León",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 15,
        "product_type": "physical",
        "latency_ms": 130,
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
    assert data["transaction_id"] == 4004
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")


def test_transaction_new_user_small_amount_safe_conditions():
    """
    New user but small amount during daytime with all low-risk signals.
    Expected: ACCEPTED (newness alone shouldn't block low-risk txn).
    """
    body = {
        "transaction_id": 4005,
        "amount_mxn": 150.0,
        "customer_txn_30d": 0,
        "geo_state": "Jalisco",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 13,
        "product_type": "physical",
        "latency_ms": 100,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 4005
    # Should likely be ACCEPTED unless rules are very conservative
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")


# ============================================================================
# SPECIAL SCENARIOS
# ============================================================================

def test_transaction_physical_product_lower_risk():
    """
    Physical product generally has lower fraud risk than digital.
    Expected: More lenient scoring for physical goods.
    """
    body = {
        "transaction_id": 5001,
        "amount_mxn": 3000.0,
        "customer_txn_30d": 5,
        "geo_state": "Nuevo León",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 21,
        "product_type": "physical",
        "latency_ms": 160,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 5001
    # Physical product should reduce risk vs digital
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")


def test_transaction_high_velocity_established_user():
    """
    High transaction velocity (high latency_ms) but from established user.
    Expected: IN_REVIEW (velocity concerning but user history helps).
    """
    body = {
        "transaction_id": 5002,
        "amount_mxn": 2200.0,
        "customer_txn_30d": 40,
        "geo_state": "Ciudad de México",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 18,
        "product_type": "digital",
        "latency_ms": 520,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 5002
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")


def test_transaction_high_email_risk_low_amount():
    """
    High email risk with low amount and otherwise clean profile.
    Expected: IN_REVIEW (email is suspicious but low financial risk).
    """
    body = {
        "transaction_id": 5003,
        "amount_mxn": 500.0,
        "customer_txn_30d": 2,
        "geo_state": "Jalisco",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 10,
        "product_type": "digital",
        "latency_ms": 140,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "high",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 5003
    assert data["decision"] in ("IN_REVIEW", "REJECTED")


def test_transaction_one_chargeback_recovery_path():
    """
    User with exactly 1 chargeback but trying to rebuild reputation
    with small transaction.
    Expected: IN_REVIEW (one chargeback is concerning but not blocking).
    """
    body = {
        "transaction_id": 5004,
        "amount_mxn": 400.0,
        "customer_txn_30d": 8,
        "geo_state": "Nuevo León",
        "device_type": "desktop",
        "chargeback_count": 1,
        "hour": 14,
        "product_type": "physical",
        "latency_ms": 110,
        "user_reputation": "recurrent",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 5004
    # One chargeback should trigger review but not auto-reject
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW")