"""Production scoring policy: model + business rules.

CONDITIONAL_GO karari uyarinca model skoru iki esikle aksiyon ureten yardimci.
`<30` yas segmentinde otomatik aksiyon yerine `manual_review` doner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

POLICY_PATH = Path("models/decision_policy.json")
RAW_PASSTHROUGH_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


@dataclass
class Decision:
    customer_id: int | str | None
    score: float
    risk_band: str  # "high" | "medium" | "low"
    action: str
    reason: str


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _age_bucket(age: float) -> str:
    if age < 30:
        return "<30"
    if age < 40:
        return "30-40"
    if age < 50:
        return "40-50"
    return "50+"


def score_and_decide(
    raw_df: pd.DataFrame,
    policy: dict[str, Any] | None = None,
    model=None,
    preprocessor=None,
) -> pd.DataFrame:
    """Ham musteri kayitlari -> skor + aksiyon DataFrame'i."""
    policy = policy or load_policy()
    model = model or joblib.load(policy["model_path"])
    preprocessor = preprocessor or joblib.load(policy["preprocessor_path"])

    missing = [c for c in RAW_PASSTHROUGH_COLS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    X = preprocessor.transform(raw_df[RAW_PASSTHROUGH_COLS])
    proba = model.predict_proba(X)[:, 1]

    high_t = policy["thresholds"]["high_risk"]
    med_t = policy["thresholds"]["medium_risk"]
    seg_rules = policy["segment_rules"].get("age_bucket", {})

    rows: list[Decision] = []
    for i, p in enumerate(proba):
        age = float(raw_df["Age"].iloc[i])
        bucket = _age_bucket(age)

        if p >= high_t:
            band, action, reason = "high", policy["actions"]["high_risk"], f"score {p:.3f} >= {high_t}"
        elif p >= med_t:
            band, action, reason = "medium", policy["actions"]["medium_risk"], f"{med_t} <= score {p:.3f} < {high_t}"
        else:
            band, action, reason = "low", policy["actions"]["low_risk"], f"score {p:.3f} < {med_t}"

        # Segment kurallari ezerek manuel inceleme
        seg_rule = seg_rules.get(bucket)
        if seg_rule and band != "low":
            action = seg_rule["action"]
            reason = f"{seg_rule['reason']} (model_band={band})"

        cust_id = raw_df["CustomerId"].iloc[i] if "CustomerId" in raw_df.columns else None
        rows.append(Decision(customer_id=cust_id, score=float(p), risk_band=band, action=action, reason=reason))

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_actions(decisions: pd.DataFrame) -> pd.DataFrame:
    return (
        decisions.groupby("action", as_index=False)
        .agg(n=("score", "size"), mean_score=("score", "mean"))
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )