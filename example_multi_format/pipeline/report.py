"""Generate final YAML report from summary."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def run():
    summary = pd.read_csv(ROOT / "data/processed/summary.csv")

    report = {
        "generated": "2026-05-19",
        "total_products": int(len(summary)),
        "total_revenue": float(summary["total_revenue"].sum()),
        "top_product": summary.sort_values("total_revenue", ascending=False).iloc[0]["product"],
    }

    with open(ROOT / "data/processed/report.yaml", "w") as f:
        yaml.dump(report, f, default_flow_style=False)

    import json
    with open(ROOT / "data/processed/metrics.json", "w") as f:
        json.dump({"metric": "revenue", "value": report["total_revenue"]}, f)

    print(f"Report: {report['total_products']} products, ${report['total_revenue']:.2f} revenue")


if __name__ == "__main__":
    run()
