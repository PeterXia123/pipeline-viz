"""Aggregate orders → summary CSV + model pickle."""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run():
    df = pd.read_parquet(ROOT / "data/processed/orders_clean.parquet")

    summary = df.groupby("product").agg(
        total_qty=("qty", "sum"),
        total_revenue=("revenue", "sum"),
        order_count=("order_id", "nunique"),
    ).reset_index()

    summary.to_csv(ROOT / "data/processed/summary.csv", index=False)

    model = {"mean_revenue": float(df["revenue"].mean()), "products": list(summary["product"])}
    with open(ROOT / "data/processed/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Summary: {len(summary)} products, model saved")


if __name__ == "__main__":
    run()
