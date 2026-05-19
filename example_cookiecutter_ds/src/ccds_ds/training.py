"""Minimal training stub for notebook 03."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_stub(df: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    """Fit a tiny classifier on engineered columns; returns (model, metrics)."""
    if "high_value" not in df.columns:
        raise ValueError("expected high_value from feature step")
    feature_cols = [c for c in df.columns if c in ("amount", "amount_log")]
    if not feature_cols:
        feature_cols = ["amount"]
    X = df[feature_cols].fillna(0).values
    y = df["high_value"].values
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    acc = float((model.predict(X) == y).mean()) if len(y) else 0.0
    metrics = {"train_accuracy": acc, "n_rows": float(len(df))}
    return model, metrics
