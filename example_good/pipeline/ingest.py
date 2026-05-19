"""扇入：两路 raw → 一路 staging。"""

import pandas as pd


def merge_sources() -> None:
    orders = pd.read_csv("data/raw/orders.csv")
    customers = pd.read_csv("data/raw/customers.csv")
    merged = orders.merge(customers, on="id", how="inner")
    merged.to_csv("data/staging/merged_orders.csv", index=False)
