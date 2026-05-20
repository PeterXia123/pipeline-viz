"""Simple customer segmentation based on RFM-like features."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def segment_users(user_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = user_features.copy()
    spend_median = df["total_spend_usd"].median()
    freq_median = df["order_count"].median()

    conditions = [
        (df["total_spend_usd"] >= spend_median) & (df["order_count"] >= freq_median),
        (df["total_spend_usd"] >= spend_median) & (df["order_count"] < freq_median),
        (df["total_spend_usd"] < spend_median) & (df["order_count"] >= freq_median),
    ]
    labels = ["vip", "high_value", "frequent", "casual"]
    df["segment"] = np.select(conditions, labels[:3], default="casual")

    centroids = {}
    for seg in df["segment"].unique():
        subset = df[df["segment"] == seg]
        centroids[seg] = {
            "avg_spend_usd": round(float(subset["total_spend_usd"].mean()), 2),
            "avg_orders": round(float(subset["order_count"].mean()), 2),
            "count": int(len(subset)),
        }
    return df, centroids
