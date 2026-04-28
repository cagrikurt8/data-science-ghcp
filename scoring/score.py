"""Azure ML online endpoint entry script.

Azure ML, AZUREML_MODEL_DIR ortam degiskeni ile model dosyalarini saglar.
- init(): bir kere yuklenir; model + preprocessor + policy memory'e alinir.
- run(raw_data): JSON string alir, JSON string doner.

Sozlesme: AzureML deployment.yml `code_configuration` ile bu dosyayi gosterir.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger("bank-churn-aml")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

_state: dict = {}


def init() -> None:
    model_dir = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    # AML registered model: model_dir/<asset>/best_model.joblib (asset name baglantili)
    candidates = list(model_dir.rglob("best_model.joblib"))
    if not candidates:
        raise FileNotFoundError(f"best_model.joblib not found under {model_dir}")
    model_path = candidates[0]
    base = model_path.parent
    _state["model"] = joblib.load(model_path)
    _state["preprocessor"] = joblib.load(base / "preprocessor.joblib")
    _state["policy"] = json.loads((base / "decision_policy.json").read_text(encoding="utf-8"))
    logger.info("AML init OK. policy=%s", _state["policy"].get("version"))


def run(raw_data: str) -> str:
    """Input contract:
    {"customers": [ {CreditScore, Geography, Gender, Age, Tenure, Balance,
                       NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                       CustomerId?}, ... ]}
    """
    try:
        from src.serve.policy import score_and_decide  # imports at runtime; src/ baked in image
    except ImportError:
        # Fallback: scoring/score.py kendi basina paketleniyorsa policy modulu kopyalanmali
        raise
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    customers = payload.get("customers", payload if isinstance(payload, list) else [])
    if not customers:
        return json.dumps({"error": "empty 'customers' list"})

    raw_df = pd.DataFrame(customers)
    decisions = score_and_decide(
        raw_df,
        policy=_state["policy"],
        model=_state["model"],
        preprocessor=_state["preprocessor"],
    )
    return json.dumps({
        "model_version": _state["policy"].get("version"),
        "decision": _state["policy"].get("decision"),
        "predictions": decisions.to_dict(orient="records"),
    })