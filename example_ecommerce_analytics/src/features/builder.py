"""Feature engineering for user and order analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_user_features(merged: pd.DataFrame) -> pd.DataFrame:
    uf = merged.groupby("user_id").agg(
        order_count=("order_id", "nunique"),
        total_spend_usd=("total_usd", "sum"),
        avg_order_usd=("total_usd", "mean"),
        unique_categories=("category", "nunique"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    ).reset_index()
    uf["spend_log"] = np.log1p(uf["total_spend_usd"])
    uf["days_active"] = (uf["last_order"] - uf["first_order"]).dt.days
    return uf


def build_order_features(merged: pd.DataFrame) -> pd.DataFrame:
    of = merged.copy()
    of["is_high_value"] = (of["total_usd"] > 200).astype(int)
    of["order_month"] = of["order_date"].dt.to_period("M").astype(str)
    monthly = of.groupby("order_month").agg(
        monthly_revenue_usd=("total_usd", "sum"),
        monthly_orders=("order_id", "nunique"),
        monthly_margin_usd=("margin_usd", "sum"),
    ).reset_index()
    return monthly
