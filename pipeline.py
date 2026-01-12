# pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from plan import Plan

TARGET_COLUMN = "Survived"

@dataclass
class RunResult:
    confusion: np.ndarray
    metrics: Dict[str, float]
    used_columns: List[str]


def _build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    # default: logistic regression
    return LogisticRegression(max_iter=3000)


def run_training(df: pd.DataFrame, plan: Plan):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' column.")

    # Split target and features
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    # Keep columns specified by the user.
    if plan.keep_only_columns:
        keep = [c for c in plan.keep_only_columns if c in X.columns]
        X = X[keep]

    # Drop the columns specified by the user.
    if plan.drop_columns:
        drop = [c for c in plan.drop_columns if c in X.columns]
        if drop:
            X = X.drop(columns=drop)

    used_columns = X.columns.tolist()

    # Detect numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=plan.numeric_impute)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=plan.categorical_impute)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    # Model
    model = _build_model(plan.model)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    return RunResult(confusion=cm, metrics=metrics, used_columns=used_columns)
