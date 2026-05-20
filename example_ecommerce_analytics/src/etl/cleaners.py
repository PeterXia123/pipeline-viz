"""Data cleaning and merge logic."""
from __future__ import annotations

import pandas as pd


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.dropna(subset=["price_cny"]).copy()
    df["total_cny"] = df["quantity"] * df["price_cny"]
    return df


def merge_order_details(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    merged = orders.merge(products, on="product_id", how="left")
    merged = merged.merge(users, on="user_id", how="left")
    return merged


def convert_currency(df: pd.DataFrame, rate: float, col: str = "total_cny") -> pd.DataFrame:
    out = df.copy()
    out["total_usd"] = (out[col] * rate).round(2)
    out["margin_usd"] = ((out["total_cny"] - out["cost_cny"] * out["quantity"]) * rate).round(2)
    return out
