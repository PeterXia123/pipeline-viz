"""Raw data loaders with basic validation."""
from __future__ import annotations

import pandas as pd


def load_orders() -> pd.DataFrame:
    df = pd.read_csv("data/raw/orders.csv", parse_dates=["order_date"])
    df["price_cny"] = pd.to_numeric(df["price_cny"], errors="coerce")
    return df


def load_products() -> pd.DataFrame:
    return pd.read_csv("data/raw/products.csv")


def load_users() -> pd.DataFrame:
    df = pd.read_csv("data/raw/users.csv", parse_dates=["signup_date"])
    return df
