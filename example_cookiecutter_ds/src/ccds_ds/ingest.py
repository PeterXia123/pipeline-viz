"""Helpers for ingest stage (optional imports from notebooks)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_geo_lookup(project_root: Path) -> pd.DataFrame:
    # 使用相对路径字符串便于 pipeline-viz 从源码解析 I/O（运行时期望 cwd 为项目根）
    _ = project_root  # 保留参数供调用方显式传入项目根；读文件用固定相对路径
    return pd.read_csv("data/external/geo_lookup.csv")


def merge_with_geo(merged_orders: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    geo = load_geo_lookup(project_root)
    # merge on region from customers — stub: attach timezone by joining region from a column if present
    if "region" not in merged_orders.columns:
        return merged_orders
    return merged_orders.merge(geo, on="region", how="left")
