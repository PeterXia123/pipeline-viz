from __future__ import annotations

import io
from pathlib import Path
from typing import Any


_DF_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".pkl", ".pickle"}


def is_dataframe_format(path: str) -> bool:
    return Path(path).suffix.lower() in _DF_SUFFIXES


def extract_profile(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": 0,
            "format": "missing",
            "error": "file not found",
        }

    file_size = file_path.stat().st_size
    suf = file_path.suffix.lower()

    if suf not in _DF_SUFFIXES:
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": file_size,
            "format": "non_dataframe",
            "error": None,
        }

    try:
        df = _read_df(file_path)
        return _profile_from_df(df, file_size, suf)
    except Exception as e:
        # pkl/pickle files that can't be read as a DataFrame are treated as
        # opaque non-dataframe objects (e.g. model weights, arbitrary objects)
        fmt = "non_dataframe" if suf in (".pkl", ".pickle") else suf.lstrip(".")
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": file_size,
            "format": fmt,
            "error": str(e)[:200],
        }


def extract_profile_from_bytes(data: bytes, suffix: str) -> dict[str, Any]:
    suf = suffix.lower()
    try:
        df = _read_df_bytes(data, suf)
        return _profile_from_df(df, len(data), suf)
    except Exception as e:
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": len(data),
            "format": suf.lstrip("."),
            "error": str(e)[:200],
        }


def _profile_from_df(df, file_size: int, suf: str) -> dict[str, Any]:
    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "file_size": file_size,
        "format": suf.lstrip("."),
        "error": None,
    }


def _read_df(path: Path):
    import pandas as pd
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"unsupported: {suf}")


def _read_df_bytes(data: bytes, suf: str):
    import pandas as pd
    bio = io.BytesIO(data)
    if suf == ".csv":
        return pd.read_csv(bio)
    if suf == ".tsv":
        return pd.read_csv(bio, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(bio)
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(bio)
    raise ValueError(f"unsupported: {suf}")


def compute_schema_diff(
    old_profile: dict[str, Any], new_profile: dict[str, Any]
) -> dict[str, Any]:
    old_cols = set(old_profile.get("columns") or [])
    new_cols = set(new_profile.get("columns") or [])
    old_dtypes = old_profile.get("dtypes") or {}
    new_dtypes = new_profile.get("dtypes") or {}

    type_changes = {}
    for col in old_cols & new_cols:
        if old_dtypes.get(col) != new_dtypes.get(col):
            type_changes[col] = {
                "old": old_dtypes.get(col),
                "new": new_dtypes.get(col),
            }

    return {
        "row_count_old": old_profile.get("row_count"),
        "row_count_new": new_profile.get("row_count"),
        "col_count_old": old_profile.get("col_count"),
        "col_count_new": new_profile.get("col_count"),
        "columns_added": sorted(new_cols - old_cols),
        "columns_removed": sorted(old_cols - new_cols),
        "type_changes": type_changes,
        "file_size_old": old_profile.get("file_size", 0),
        "file_size_new": new_profile.get("file_size", 0),
    }
