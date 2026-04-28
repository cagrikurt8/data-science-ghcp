"""Feature engineering pipeline for the Churn_Modelling dataset.

Tasarim notlari:
- Veri Analyzer raporu (`reports/data_quality.md`) baz alindi.
- PII (`Surname`) ve kimlik kolonlari (`RowNumber`, `CustomerId`) drop edilir.
- Turetilen ozellikler hedef sizintisi olusturmaz; yalnizca girdi feature'larindan
    hesaplanir (Data Analyzer R4 onerisi: `has_balance`, `balance_to_salary` vb.).
- Sayisal: median impute + StandardScaler.
- Kategorik: mode impute + OneHotEncoder (low cardinality: Geography, Gender).
- Train/val/test stratified split, `fit` yalnizca train'de yapilir (sizinti yok).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

NUMERIC_RAW = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "HasCrCard",
    "IsActiveMember",
]
CATEGORICAL = ["Geography", "Gender"]
DERIVED_NUMERIC = [
    "has_balance",
    "balance_to_salary",
    "products_per_tenure",
    "is_senior",
]


@dataclass
class SplitPaths:
    train: Path
    val: Path
    test: Path


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Hedef sizintisi olusturmayan turetilmis feature'lar uretir.

    - has_balance: Balance > 0 (Data Analyzer R4)
    - balance_to_salary: Balance / (EstimatedSalary + 1)
    - products_per_tenure: NumOfProducts / (Tenure + 1)
    - is_senior: Age >= 60 (Data Analyzer R6 - sag carpik yas dagilimi)
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        X["has_balance"] = (X["Balance"] > 0).astype(int)
        X["balance_to_salary"] = X["Balance"] / (X["EstimatedSalary"] + 1.0)
        X["products_per_tenure"] = X["NumOfProducts"] / (X["Tenure"] + 1.0)
        X["is_senior"] = (X["Age"] >= 60).astype(int)
        return X


def build_preprocessor() -> Pipeline:
    """ColumnTransformer + Pipeline; fit yalnizca train uzerinde cagrilmali."""
    numeric_features = NUMERIC_RAW + DERIVED_NUMERIC

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("derived", DerivedFeatures()),
            ("preprocess", column_transformer),
        ]
    )


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split (sizinti yok: yalnizca satir bolme)."""
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df[target],
        random_state=random_state,
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df[target],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def transform_to_frame(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Donusturulmus matrisi feature isimleriyle DataFrame'e cevirir."""
    X = df.drop(columns=[TARGET_COL])
    matrix = pipeline.transform(X)
    out = pd.DataFrame(matrix, columns=feature_names, index=df.index)
    out[TARGET_COL] = df[TARGET_COL].to_numpy()
    return out


def run(
    raw_csv: Path = Path("data/raw/Churn_Modelling.csv"),
    out_dir: Path = Path("data/processed"),
    model_path: Path = Path("models/preprocessor.joblib"),
) -> dict[str, object]:
    """Tam pipeline: yukle -> mask -> split -> fit -> transform -> kaydet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    train_df, val_df, test_df = stratified_split(df)

    pipeline = build_preprocessor()
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    pipeline.fit(X_train, y_train)

    feature_names = list(
        pipeline.named_steps["preprocess"].get_feature_names_out()
    )

    train_out = transform_to_frame(pipeline, train_df, feature_names)
    val_out = transform_to_frame(pipeline, val_df, feature_names)
    test_out = transform_to_frame(pipeline, test_df, feature_names)

    paths = SplitPaths(
        train=out_dir / "train.parquet",
        val=out_dir / "val.parquet",
        test=out_dir / "test.parquet",
    )
    train_out.to_parquet(paths.train, index=False)
    val_out.to_parquet(paths.val, index=False)
    test_out.to_parquet(paths.test, index=False)

    joblib.dump(pipeline, model_path)

    return {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "shapes": {
            "train": train_out.shape,
            "val": val_out.shape,
            "test": test_out.shape,
        },
        "target_ratio": {
            "train": float(train_df[TARGET_COL].mean()),
            "val": float(val_df[TARGET_COL].mean()),
            "test": float(test_df[TARGET_COL].mean()),
        },
        "paths": paths,
        "model_path": model_path,
    }


