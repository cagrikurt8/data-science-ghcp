"""Training pipeline for bank churn classification.

- Baseline: LogisticRegression
- Strong model: XGBoost (with Optuna HPO, 20 trials, target=PR-AUC)
- MLflow experiment: bank-churn (params + metrics + model logged per run)
- Validation set used for early stopping; test set untouched until final eval
- random_state=42 throughout
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

RANDOM_STATE = 42
TARGET_COL = "Exited"
EXPERIMENT_NAME = "bank-churn"
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
N_TRIALS = 20


@dataclass
class Datasets:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    sample_weight_train: np.ndarray | None = None


def _age_bucket(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 40, 50, 200], labels=["<30", "30-40", "40-50", "50+"], right=False).astype(str)


def compute_fairness_weights(raw_train: pd.DataFrame) -> np.ndarray:
    """age_bucket x Geography strata uzerinden inverse-frequency sample_weight.

    v1 evaluation'da `<30` recall 0.14 ve Geography Fransa recall 0.67 idi.
    Bu segmentleri agırlıklandırarak modeli dengelemeyi hedefliyoruz.
    Agirliklar normalize edilir (ortalama 1).
    """
    strata = _age_bucket(raw_train["Age"]) + "|" + raw_train["Geography"].astype(str)
    counts = strata.value_counts()
    inv = 1.0 / counts
    weights = strata.map(inv).to_numpy(dtype=float, copy=True)
    weights = weights * (len(weights) / weights.sum())  # mean=1
    return weights


def load_splits(processed_dir: Path = PROCESSED_DIR) -> Datasets:
    """Feature-engineered parquet'leri yukle."""
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df.drop(columns=[TARGET_COL]), df[TARGET_COL].astype(int)

    X_train, y_train = split(train)
    X_val, y_val = split(val)
    X_test, y_test = split(test)
    return Datasets(X_train, y_train, X_val, y_val, X_test, y_test)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Dengesiz sinif icin PR-AUC oncelikli metrik seti."""
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
    }


def train_baseline(data: Datasets) -> tuple[LogisticRegression, dict[str, float]]:
    """LogisticRegression baseline. Sinif dengesizligi icin class_weight=balanced."""
    with mlflow.start_run(run_name="baseline_logreg"):
        params = {
            "model": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)

        clf = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            random_state=RANDOM_STATE,
        )
        clf.fit(data.X_train, data.y_train)

        val_metrics = compute_metrics(
            data.y_val.to_numpy(), clf.predict_proba(data.X_val)[:, 1]
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        return clf, val_metrics


def _xgb_objective(trial: optuna.Trial, data: Datasets, scale_pos_weight: float) -> float:
    """Optuna objective: PR-AUC on validation set with early stopping."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 30,
    }

    with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
        mlflow.log_params({k: v for k, v in params.items() if k != "early_stopping_rounds"})
        mlflow.log_param("trial_number", trial.number)

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            data.X_train,
            data.y_train,
            sample_weight=data.sample_weight_train,
            eval_set=[(data.X_val, data.y_val)],
            verbose=False,
        )
        y_proba = clf.predict_proba(data.X_val)[:, 1]
        metrics = compute_metrics(data.y_val.to_numpy(), y_proba)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("best_iteration", float(clf.best_iteration))
        return metrics["pr_auc"]


def train_xgboost(data: Datasets, n_trials: int = N_TRIALS) -> tuple[xgb.XGBClassifier, dict[str, float], dict[str, Any]]:
    """Optuna ile XGBoost HPO; en iyi parametrelerle final fit."""
    pos = float((data.y_train == 1).sum())
    neg = float((data.y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1.0)

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb-prauc")

    with mlflow.start_run(run_name="xgb_hpo"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        study.optimize(
            lambda t: _xgb_objective(t, data, scale_pos_weight),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        mlflow.log_metric("best_val_pr_auc", float(study.best_value))
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_estimators": 1000,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 30,
        **study.best_params,
    }

    with mlflow.start_run(run_name="xgb_final"):
        mlflow.log_params({k: v for k, v in best_params.items() if k != "early_stopping_rounds"})
        clf = xgb.XGBClassifier(**best_params)
        clf.fit(
            data.X_train,
            data.y_train,
            sample_weight=data.sample_weight_train,
            eval_set=[(data.X_val, data.y_val)],
            verbose=False,
        )
        val_metrics_raw = compute_metrics(
            data.y_val.to_numpy(), clf.predict_proba(data.X_val)[:, 1]
        )
        mlflow.log_metrics({f"val_raw_{k}": v for k, v in val_metrics_raw.items()})
        mlflow.log_metric("best_iteration", float(clf.best_iteration))

        # Isotonic calibration on validation set with FrozenEstimator
        # ModelEvaluator v1 ECE=0.189 -> hedef <0.05
        calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        calibrated.fit(data.X_val, data.y_val)
        val_metrics = compute_metrics(
            data.y_val.to_numpy(), calibrated.predict_proba(data.X_val)[:, 1]
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_param("calibration", "isotonic_prefit")

    return calibrated, val_metrics, study.best_params


def evaluate_on_test(model: Any, data: Datasets, name: str) -> dict[str, float]:
    """Test set degerlendirmesi (yalnizca finalde)."""
    proba = model.predict_proba(data.X_test)[:, 1]
    metrics = compute_metrics(data.y_test.to_numpy(), proba)
    with mlflow.start_run(run_name=f"{name}_test_eval"):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_param("model", name)
    return metrics


def run(n_trials: int = N_TRIALS) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{(Path.cwd() / 'mlruns').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_splits()

    # Fairness sample weights from raw train split (age_bucket x Geography)
    from src.features.pipeline import stratified_split, DROP_COLS as RAW_DROP
    raw = pd.read_csv("data/raw/Churn_Modelling.csv")
    raw = raw.drop(columns=[c for c in RAW_DROP if c in raw.columns])
    raw_train, _, _ = stratified_split(raw)
    data.sample_weight_train = compute_fairness_weights(raw_train)

    baseline_model, baseline_val = train_baseline(data)
    xgb_model, xgb_val, best_params = train_xgboost(data, n_trials=n_trials)

    candidates = {
        "baseline_logreg": (baseline_model, baseline_val),
        "xgboost": (xgb_model, xgb_val),
    }
    best_name = max(candidates, key=lambda k: candidates[k][1]["pr_auc"])
    best_model, best_val = candidates[best_name]

    test_metrics = evaluate_on_test(best_model, data, best_name)

    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)

    metrics_payload = {
        "best_model": best_name,
        "best_xgb_params": best_params,
        "validation": {
            "baseline_logreg": baseline_val,
            "xgboost": xgb_val,
        },
        "test": {best_name: test_metrics},
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    return {
        "best_name": best_name,
        "best_path": best_path,
        "metrics": metrics_payload,
    }