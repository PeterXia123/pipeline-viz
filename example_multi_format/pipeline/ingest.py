"""Read raw CSV + JSON config, produce staging parquet."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run():
    orders = pd.read_csv(ROOT / "data/raw/orders.csv")
    with open(ROOT / "data/config/params.json") as f:
        params = json.load(f)

    if params.get("filter_status"):
        orders = orders[orders["status"] == params["filter_status"]]

    min_qty = params.get("min_qty", 0)
    if min_qty:
        orders = orders[orders["qty"] >= min_qty]

    orders.to_parquet(ROOT / "data/processed/orders_clean.parquet", index=False)
    print(f"Ingested {len(orders)} rows")


if __name__ == "__main__":
    run()
