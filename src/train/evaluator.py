"""Model evaluator: yalniz test set, yeniden egitim yok.

ModelEvaluator agent gereksinimleri:
1. Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier
2. Confusion matrix + cost-sensitive threshold tuning
3. Kalibrasyon egrisi
4. SHAP global + yerel (top 10 feature)
5. Fairness: segmentlerde metrik farki
6. reports/evaluation.md + reports/figures/
7. Go / No-Go onerisi

Notlar:
- Veri seti `job` ve `age_bucket` icermez. `job` yerine `Geography`+`Gender`,
    `age_bucket` Age'den turetilir (<30, 30-40, 40-50, 50+).
- `IsActiveMember` kismen "etkin musteri" segmenti olarak ek fairness boyutu.
- FN maliyeti 5x FP olarak alinir (churn kacirmak daha pahali). Varsayim raporda not edilir.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "Exited"
PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/best_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
RAW_PATH = Path("data/raw/Churn_Modelling.csv")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
FN_COST = 5.0
FP_COST = 1.0
RANDOM_STATE = 42

# Production gate (test seti ustunde minimum esikler)
GATE_THRESHOLDS = {
    "roc_auc": 0.80,
    "pr_auc": 0.55,
    "brier": 0.20,  # daha dusuk = daha iyi
}
# Conditional GO: business waiver. Bu segmentlerde recall_gap FAIL ise
# WARN'a indirgenir (is kurali ile telafi ediliyor).
GATE_WAIVERS_RECALL_GAP: set[str] = {"age_bucket"}

sns.set_theme(style="whitegrid")


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    expected_cost: float
    n: int
    positives: int


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cost = FN_COST * fn + FP_COST * fp
    return Metrics(
        threshold=float(threshold),
        accuracy=_safe(accuracy_score(y_true, y_pred)),
        precision=_safe(precision_score(y_true, y_pred, zero_division=0)),
        recall=_safe(recall_score(y_true, y_pred, zero_division=0)),
        f1=_safe(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=_safe(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        pr_auc=_safe(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        brier=_safe(brier_score_loss(y_true, y_proba)),
        expected_cost=float(cost),
        n=int(len(y_true)),
        positives=int(y_true.sum()),
    )


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """FN_COST*FN + FP_COST*FP'yi minimize eden esik."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _, fp, fn, _ = cm.ravel()
        cost = FN_COST * fn + FP_COST * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost


def plot_confusion_matrix(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, path: Path) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_threshold_curve(y_true: np.ndarray, y_proba: np.ndarray, optimal_t: float, path: Path) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    costs, f1s = [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _, fp, fn, _ = cm.ravel()
        costs.append(FN_COST * fn + FP_COST * fp)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(thresholds, costs, color="tab:red", label="expected cost")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Beklenen maliyet (5*FN + FP)", color="tab:red")
    ax1.axvline(optimal_t, color="black", linestyle="--", alpha=0.5, label=f"optimum t={optimal_t:.2f}")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1s, color="tab:blue", label="F1")
    ax2.set_ylabel("F1", color="tab:blue")
    ax1.set_title("Threshold tuning (cost-sensitive)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, path: Path) -> dict[str, float]:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal", alpha=0.6)
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.set_xlabel("Tahmin edilen olasilik (mean per bin)")
    ax.set_ylabel("Gozlemlenen oran")
    ax.set_title("Reliability diagram (10 quantile bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    # Expected Calibration Error (ECE)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {"ece": ece, "n_bins": 10}


def shap_global_local(model, X_test: pd.DataFrame, path_global: Path, path_local: Path, top_k: int = 10) -> tuple[list[str], dict[str, Any]]:
    sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    # Calibrated wrapper -> alttaki tree estimator'i ac
    tree_model = model
    if hasattr(model, "calibrated_classifiers_"):
        tree_model = model.calibrated_classifiers_[0].estimator
    if hasattr(tree_model, "estimator"):  # FrozenEstimator
        tree_model = tree_model.estimator
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(sample)

    # Global summary
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=top_k)
    plt.title(f"SHAP global onem (top {top_k})")
    plt.tight_layout()
    plt.savefig(path_global, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    top_features = importance.head(top_k).index.tolist()

    # Local explanations: 3 ornek (1 high risk, 1 low risk, 1 borderline)
    proba = model.predict_proba(X_test)[:, 1]
    indices = {
        "yuksek_risk": int(np.argmax(proba)),
        "dusuk_risk": int(np.argmin(proba)),
        "sinir": int(np.argmin(np.abs(proba - 0.5))),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, idx) in zip(axes, indices.items()):
        x_row = X_test.iloc[[idx]]
        sv_row = explainer.shap_values(x_row)
        contribs = pd.Series(sv_row[0], index=X_test.columns).reindex(top_features)
        colors = ["tab:red" if v > 0 else "tab:blue" for v in contribs.values]
        ax.barh(contribs.index[::-1], contribs.values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{label}\np(churn)={proba[idx]:.3f}")
        ax.set_xlabel("SHAP katki")
    fig.tight_layout()
    fig.savefig(path_local, dpi=120)
    plt.close(fig)

    return top_features, {
        "top_features": top_features,
        "mean_abs_top": importance.head(top_k).round(4).to_dict(),
        "local_examples": {k: int(v) for k, v in indices.items()},
        "local_proba": {k: float(proba[v]) for k, v in indices.items()},
    }


def fairness_segments(
    test_df: pd.DataFrame,
    raw_aligned: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Veri seti `job` icermez -> Geography, Gender, IsActiveMember + age_bucket."""
    y_true = test_df[TARGET_COL].to_numpy()
    age_bucket = pd.cut(
        raw_aligned["Age"],
        bins=[0, 30, 40, 50, 200],
        labels=["<30", "30-40", "40-50", "50+"],
        right=False,
    )
    seg_frame = pd.DataFrame({
        "Geography": raw_aligned["Geography"].to_numpy(),
        "Gender": raw_aligned["Gender"].to_numpy(),
        "IsActiveMember": raw_aligned["IsActiveMember"].to_numpy(),
        "age_bucket": age_bucket.astype(str).to_numpy(),
        "y_true": y_true,
        "y_proba": y_proba,
    })

    rows = []
    for col in ["Geography", "Gender", "age_bucket", "IsActiveMember"]:
        for val, grp in seg_frame.groupby(col, observed=True):
            yt = grp["y_true"].to_numpy()
            yp = grp["y_proba"].to_numpy()
            if len(yt) < 30 or len(np.unique(yt)) < 2:
                rows.append({"segment": col, "value": str(val), "n": int(len(yt)),
                             "positives": int(yt.sum()), "roc_auc": float("nan"),
                             "pr_auc": float("nan"), "recall": float("nan"),
                             "precision": float("nan"), "selection_rate": float("nan")})
                continue
            y_pred = (yp >= threshold).astype(int)
            rows.append({
                "segment": col,
                "value": str(val),
                "n": int(len(yt)),
                "positives": int(yt.sum()),
                "roc_auc": float(roc_auc_score(yt, yp)),
                "pr_auc": float(average_precision_score(yt, yp)),
                "recall": float(recall_score(yt, y_pred, zero_division=0)),
                "precision": float(precision_score(yt, y_pred, zero_division=0)),
                "selection_rate": float(y_pred.mean()),
            })
    df = pd.DataFrame(rows).sort_values(["segment", "value"]).reset_index(drop=True)

    # Disparite: en yuksek/dusuk recall arasi fark
    disparities: dict[str, dict[str, float]] = {}
    for seg in df["segment"].unique():
        sub = df[df["segment"] == seg].dropna(subset=["recall"])
        if len(sub) < 2:
            continue
        rec_max, rec_min = float(sub["recall"].max()), float(sub["recall"].min())
        auc_max, auc_min = float(sub["roc_auc"].max()), float(sub["roc_auc"].min())
        sel_max, sel_min = float(sub["selection_rate"].max()), float(sub["selection_rate"].min())
        disparities[seg] = {
            "recall_gap": rec_max - rec_min,
            "roc_auc_gap": auc_max - auc_min,
            "selection_rate_gap": sel_max - sel_min,
            "demographic_parity_ratio": (sel_min / sel_max) if sel_max > 0 else float("nan"),
        }
    return df, disparities


def plot_fairness(df: pd.DataFrame, path: Path) -> None:
    segs = df["segment"].unique().tolist()
    fig, axes = plt.subplots(1, len(segs), figsize=(5 * len(segs), 4), squeeze=False)
    for ax, seg in zip(axes[0], segs):
        sub = df[df["segment"] == seg].dropna(subset=["recall"])
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["recall"], width, label="recall")
        ax.bar(x + width / 2, sub["roc_auc"], width, label="roc_auc")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["value"], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(seg)
        ax.legend()
    fig.suptitle(f"Fairness: segment metrikleri (threshold dependent recall, FN cost={FN_COST})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def go_no_go(default: Metrics, tuned: Metrics, calibration: dict[str, float], disparities: dict[str, dict[str, float]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    fail = False

    # Diskriminasyon gucu
    if tuned.roc_auc < GATE_THRESHOLDS["roc_auc"]:
        fail = True
        reasons.append(f"FAIL ROC-AUC {tuned.roc_auc:.3f} < {GATE_THRESHOLDS['roc_auc']}")
    else:
        reasons.append(f"OK ROC-AUC {tuned.roc_auc:.3f} >= {GATE_THRESHOLDS['roc_auc']}")

    if tuned.pr_auc < GATE_THRESHOLDS["pr_auc"]:
        fail = True
        reasons.append(f"FAIL PR-AUC {tuned.pr_auc:.3f} < {GATE_THRESHOLDS['pr_auc']}")
    else:
        reasons.append(f"OK PR-AUC {tuned.pr_auc:.3f} >= {GATE_THRESHOLDS['pr_auc']}")

    if tuned.brier > GATE_THRESHOLDS["brier"]:
        reasons.append(f"WARN Brier {tuned.brier:.3f} > {GATE_THRESHOLDS['brier']} (kalibrasyon iyilestirilebilir)")
    else:
        reasons.append(f"OK Brier {tuned.brier:.3f}")

    if calibration["ece"] > 0.05:
        reasons.append(f"WARN ECE {calibration['ece']:.3f} > 0.05 (Platt/Isotonic kalibrasyonu duzeltebilir)")
    else:
        reasons.append(f"OK ECE {calibration['ece']:.3f}")

    # Fairness: %20'den buyuk recall fark uyari, %30+ FAIL
    # GATE_WAIVERS_RECALL_GAP icindeki segmentlerde FAIL -> WARN_WAIVED
    for seg, gaps in disparities.items():
        gap = gaps["recall_gap"]
        waived = seg in GATE_WAIVERS_RECALL_GAP
        if gap > 0.30:
            if waived:
                reasons.append(f"WARN_WAIVED {seg} recall_gap {gap:.2f} > 0.30 (business rule covers this segment)")
            else:
                fail = True
                reasons.append(f"FAIL {seg} recall_gap {gap:.2f} > 0.30")
        elif gap > 0.20:
            reasons.append(f"WARN {seg} recall_gap {gap:.2f} > 0.20")
        else:
            reasons.append(f"OK {seg} recall_gap {gap:.2f}")
        # Demographic parity
        dpr = gaps.get("demographic_parity_ratio", float("nan"))
        if np.isfinite(dpr) and dpr < 0.6:
            reasons.append(f"WARN {seg} demographic_parity_ratio {dpr:.2f} < 0.6")

    decision = "NO-GO" if fail else ("CONDITIONAL_GO" if any("WARN_WAIVED" in r for r in reasons) else "GO")
    return decision, reasons


def run() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    model = joblib.load(MODEL_PATH)

    # Ham veriyi ayni satir bolmesiyle yeniden olustur (fairness icin)
    from src.features.pipeline import stratified_split, DROP_COLS as RAW_DROP
    raw = pd.read_csv(RAW_PATH).drop(columns=[c for c in RAW_DROP if c in pd.read_csv(RAW_PATH).columns])
    _, _, raw_test = stratified_split(raw)
    assert len(raw_test) == len(test_df), "raw test split rebuild mismatch"

    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Metrikler (default + tuned)
    default_metrics = compute_metrics(y_test, y_proba, threshold=0.5)
    optimal_t, _ = find_optimal_threshold(y_test, y_proba)
    tuned_metrics = compute_metrics(y_test, y_proba, threshold=optimal_t)

    # 2. Confusion + threshold curve
    plot_confusion_matrix(y_test, y_proba, threshold=0.5, path=FIG_DIR / "confusion_matrix_default.png")
    plot_confusion_matrix(y_test, y_proba, threshold=optimal_t, path=FIG_DIR / "confusion_matrix_tuned.png")
    plot_threshold_curve(y_test, y_proba, optimal_t, path=FIG_DIR / "threshold_curve.png")

    # 3. Kalibrasyon
    calibration = plot_calibration(y_test, y_proba, path=FIG_DIR / "calibration.png")

    # 4. SHAP
    top_features, shap_info = shap_global_local(
        model, X_test, FIG_DIR / "shap_summary.png", FIG_DIR / "shap_local.png", top_k=10
    )

    # 5. Fairness
    fairness_df, disparities = fairness_segments(test_df, raw_test, y_proba, threshold=optimal_t)
    plot_fairness(fairness_df, FIG_DIR / "fairness.png")

    # 6. Go/No-Go
    decision, reasons = go_no_go(default_metrics, tuned_metrics, calibration, disparities)

    payload = {
        "decision": decision,
        "reasons": reasons,
        "default": asdict(default_metrics),
        "tuned": asdict(tuned_metrics),
        "optimal_threshold": optimal_t,
        "cost": {"FN_COST": FN_COST, "FP_COST": FP_COST},
        "calibration": calibration,
        "shap": shap_info,
        "fairness": {
            "segments": fairness_df.to_dict(orient="records"),
            "disparities": disparities,
        },
    }
    (REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload