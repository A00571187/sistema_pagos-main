import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    import os as _os
    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass

def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5

def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


@dataclass
class ScoreBuilder:
    score: int = 0
    reasons: Optional[List[str]] = field(default_factory=list)

    def add(self, points: int, reason: str):
        if points != 0:
            self.score += points
            sign = "+" if points > 0 else ""
            self.reasons.append(f"{reason}({sign}{points})")

    def text_reasons(self) -> str:
        return ";".join(self.reasons)


# === Helpers de lectura segura ===============================================
def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def _as_lower_str(value, default="") -> str:
    return str(value if value is not None else default).lower()

def _as_upper_str(value, default="") -> str:
    return str(value if value is not None else default).upper()


# === Reglas atómicas ==========================================================
def _hard_block(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    chargebacks = _as_int(row.get("chargeback_count", 0))
    ip_high = _as_lower_str(row.get("ip_risk", "low")) == "high"
    return chargebacks >= cfg["chargeback_hard_block"] and ip_high

def _apply_categorical_risks(sb: ScoreBuilder, row: pd.Series, cfg: Dict[str, Any]):
    for field, mapping in [
        ("ip_risk", cfg["score_weights"]["ip_risk"]),
        ("email_risk", cfg["score_weights"]["email_risk"]),
        ("device_fingerprint_risk", cfg["score_weights"]["device_fingerprint_risk"]),
    ]:
        val = _as_lower_str(row.get(field, "low"), "low")
        add = mapping.get(val, 0)
        if add:
            sb.add(add, f"{field}:{val}")

def _apply_user_reputation(sb: ScoreBuilder, rep: str, cfg: Dict[str, Any]):
    rep_add = cfg["score_weights"]["user_reputation"].get(rep, 0)
    if rep_add:
        # conservar el formato original con signo cuando es negativo o positivo
        sb.reasons.append(f"user_reputation:{rep}({('+' if rep_add >= 0 else '')}{rep_add})")
        sb.score += rep_add

def _apply_night_hour(sb: ScoreBuilder, hour: int, cfg: Dict[str, Any]):
    if is_night(hour):
        sb.add(cfg["score_weights"]["night_hour"], f"night_hour:{hour}")

def _apply_geo_mismatch(sb: ScoreBuilder, bin_c: str, ip_c: str, cfg: Dict[str, Any]):
    if bin_c and ip_c and bin_c != ip_c:
        sb.add(cfg["score_weights"]["geo_mismatch"], f"geo_mismatch:{bin_c}!={ip_c}")

def _apply_amount_and_newuser(sb: ScoreBuilder, amount: float, ptype: str, rep: str, cfg: Dict[str, Any]):
    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        sb.add(add, f"high_amount:{ptype}:{amount}")
        if rep == "new":
            add2 = cfg["score_weights"]["new_user_high_amount"]
            sb.add(add2, "new_user_high_amount")

def _apply_latency_extreme(sb: ScoreBuilder, latency_ms: int, cfg: Dict[str, Any]):
    if latency_ms >= cfg["latency_ms_extreme"]:
        sb.add(cfg["score_weights"]["latency_extreme"], f"latency_extreme:{latency_ms}ms")

def _apply_frequency_buffer(sb: ScoreBuilder, rep: str, freq_30d: int):
    if rep in ("recurrent", "trusted") and freq_30d >= 3 and sb.score > 0:
        # mantener literal el texto original
        sb.score -= 1
        sb.reasons.append("frequency_buffer(-1)")

def _map_score_to_decision(score: int, cfg: Dict[str, Any]) -> str:
    if score >= cfg["score_to_decision"]["reject_at"]:
        return DECISION_REJECTED
    if score >= cfg["score_to_decision"]["review_at"]:
        return DECISION_IN_REVIEW
    return DECISION_ACCEPTED


# === Función principal (refactor) ============================================
def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Hard block early-return
    if _hard_block(row, cfg):
        return {
            "decision": DECISION_REJECTED,
            "risk_score": 100,
            "reasons": "hard_block:chargebacks>=2+ip_high",
        }

    sb = ScoreBuilder()

    # 2) Signals pre-normalizadas
    rep = _as_lower_str(row.get("user_reputation", "new"), "new")
    hour = _as_int(row.get("hour", 12), 12)
    bin_c = _as_upper_str(row.get("bin_country", ""), "")
    ip_c = _as_upper_str(row.get("ip_country", ""), "")
    amount = _as_float(row.get("amount_mxn", 0.0), 0.0)
    ptype = _as_lower_str(row.get("product_type", "_default"), "_default")
    latency_ms = _as_int(row.get("latency_ms", 0), 0)
    freq_30d = _as_int(row.get("customer_txn_30d", 0), 0)

    # 3) Reglas
    _apply_categorical_risks(sb, row, cfg)
    _apply_user_reputation(sb, rep, cfg)
    _apply_night_hour(sb, hour, cfg)
    _apply_geo_mismatch(sb, bin_c, ip_c, cfg)
    _apply_amount_and_newuser(sb, amount, ptype, rep, cfg)
    _apply_latency_extreme(sb, latency_ms, cfg)
    _apply_frequency_buffer(sb, rep, freq_30d)

    # 4) Decisión
    decision = _map_score_to_decision(sb.score, cfg)
    return {"decision": decision, "risk_score": int(sb.score), "reasons": sb.text_reasons()}


def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = assess_row(row, cfg)
        results.append(res)
    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]
    out.to_csv(output_csv, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()