"""Shared I/O helpers used by multiple notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(data: dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_exchange_rate(currency: str = "CNY") -> float:
    rates = pd.read_csv("data/external/exchange_rates.csv")
    row = rates[rates["currency"] == currency]
    if row.empty:
        return 1.0
    return float(row.iloc[0]["rate_to_usd"])
