from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import datacompy as dc
except ImportError:  # pragma: no cover
    dc = None  # type: ignore


def _read_df_from_path(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise ValueError("读取 parquet 需安装 pyarrow：pip install pyarrow") from e
        except Exception as e:
            raise ValueError(f"无法读取 parquet：{e}") from e
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"不支持的 DataFrame 文件格式: {path.suffix}")


def _read_df_from_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    suf = suffix.lower()
    bio = io.BytesIO(data)
    if suf == ".csv":
        return pd.read_csv(bio)
    if suf == ".tsv":
        return pd.read_csv(bio, sep="\t")
    if suf in (".parquet", ".pq"):
        try:
            return pd.read_parquet(bio)
        except ImportError as e:
            raise ValueError("读取 parquet 需安装 pyarrow：pip install pyarrow") from e
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(bio)
    raise ValueError(f"不支持的 DataFrame 文件格式: {suffix}")


def normalize_merge_keys(keys: list[str] | None) -> list[str]:
    if not keys:
        return []
    return [k.strip() for k in keys if k and str(k).strip()]


def align_merge_key_types(df1: pd.DataFrame, df2: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将 join 列尽量对齐为可比较的同类型（数值优先，否则转字符串）。"""
    d1 = df1.copy()
    d2 = df2.copy()
    for k in keys:
        if k not in d1.columns or k not in d2.columns:
            continue
        t1, t2 = d1[k].dtype, d2[k].dtype
        if t1 == t2:
            continue
        try:
            n1 = pd.to_numeric(d1[k], errors="coerce")
            n2 = pd.to_numeric(d2[k], errors="coerce")
            if n1.notna().sum() > 0 and n2.notna().sum() > 0:
                d1[k], d2[k] = n1, n2
                continue
        except Exception:
            pass
        d1[k] = d1[k].astype(str)
        d2[k] = d2[k].astype(str)
    return d1, d2


def run_datacompy_compare(
    left_path: Path,
    right_path: Path,
    merge_keys: list[str] | None,
) -> dict[str, Any]:
    """
    left = 旧版（通常为快照中的文件）, right = 新版（通常为磁盘当前文件）。
    merge_keys 为空或仅空白：按行号对齐比较。
    """
    if dc is None:
        return {"error": "datacompy 未安装", "report": "", "matches": None}

    df1 = _read_df_from_path(left_path)
    df2 = _read_df_from_path(right_path)
    keys = normalize_merge_keys(merge_keys)

    if not keys:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        df1["_pipeline_row__"] = range(len(df1))
        df2["_pipeline_row__"] = range(len(df2))
        join_cols = ["_pipeline_row__"]
    else:
        missing = [k for k in keys if k not in df1.columns or k not in df2.columns]
        if missing:
            return {
                "error": f"merge key 在某一侧不存在: {missing}",
                "report": "",
                "matches": False,
            }
        df1, df2 = align_merge_key_types(df1, df2, keys)
        join_cols = keys

    try:
        compare = dc.Compare(df1, df2, join_columns=join_cols, abs_tol=0, rel_tol=0)
        report = compare.report()
        return {
            "report": report if isinstance(report, str) else str(report),
            "matches": bool(compare.matches()),
            "join_columns": join_cols,
        }
    except Exception as e:  # pragma: no cover
        return {"error": str(e), "report": "", "matches": False}


def compare_snapshot_blob_to_disk(
    snapshot_bytes: bytes,
    rel_path: str,
    disk_path: Path,
    merge_keys: list[str] | None,
) -> dict[str, Any]:
    """将快照中的字节与磁盘文件做 DataComPy 比较。"""
    suf = Path(rel_path).suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        left = Path(td) / ("snapshot" + suf)
        left.write_bytes(snapshot_bytes)
        return run_datacompy_compare(left, disk_path, merge_keys)


def is_dataframe_file(path: str) -> bool:
    suf = Path(path).suffix.lower()
    return suf in (".csv", ".tsv", ".parquet", ".pq", ".pkl", ".pickle")
