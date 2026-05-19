"""由 summary 产出指标 JSON。"""

import json

import pandas as pd


def write_metrics() -> None:
    s = pd.read_csv("data/out/summary.csv")
    payload = {"rows": int(len(s)), "version": 1}
    with open("data/out/metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
