from pathlib import Path
import json

from pipeline_viz.data_profile import (
    extract_profile,
    extract_profile_from_bytes,
    is_dataframe_format,
    compute_schema_diff,
)


def test_csv_profile(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("id,name,amount\n1,alice,10.5\n2,bob,20.3\n", encoding="utf-8")
    p = extract_profile(f)
    assert p["row_count"] == 2
    assert p["col_count"] == 3
    assert p["columns"] == ["id", "name", "amount"]
    assert p["file_size"] > 0
    assert "dtypes" in p
    assert p["dtypes"]["id"] is not None


def test_tsv_profile(tmp_path: Path):
    f = tmp_path / "data.tsv"
    f.write_text("a\tb\n1\t2\n3\t4\n5\t6\n", encoding="utf-8")
    p = extract_profile(f)
    assert p["row_count"] == 3
    assert p["col_count"] == 2


def test_parquet_profile(tmp_path: Path):
    import pandas as pd
    f = tmp_path / "data.parquet"
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_parquet(f)
    p = extract_profile(f)
    assert p["row_count"] == 2
    assert "x" in p["columns"]
    assert "y" in p["columns"]


def test_non_dataframe_profile(tmp_path: Path):
    f = tmp_path / "model.pkl"
    f.write_bytes(b"\x80\x05\x95")
    p = extract_profile(f)
    assert p["row_count"] is None
    assert p["col_count"] is None
    assert p["file_size"] > 0
    assert p["format"] == "non_dataframe"


def test_missing_file(tmp_path: Path):
    p = extract_profile(tmp_path / "nope.csv")
    assert p["row_count"] is None
    assert p["file_size"] == 0
    assert p["error"] is not None


def test_is_dataframe_format():
    assert is_dataframe_format("data.csv") is True
    assert is_dataframe_format("data.tsv") is True
    assert is_dataframe_format("data.parquet") is True
    assert is_dataframe_format("data.pq") is True
    assert is_dataframe_format("model.pkl") is True
    assert is_dataframe_format("config.yaml") is False
    assert is_dataframe_format("model.h5") is False
    assert is_dataframe_format("data.npz") is False


def test_profile_from_bytes(tmp_path: Path):
    data = b"a,b\n1,2\n3,4\n"
    p = extract_profile_from_bytes(data, ".csv")
    assert p["row_count"] == 2
    assert p["col_count"] == 2


# --- compute_schema_diff tests ---

def _make_profile(columns, dtypes, row_count=10, file_size=100):
    return {
        "row_count": row_count,
        "col_count": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "file_size": file_size,
        "format": "csv",
        "error": None,
    }


def test_schema_diff_columns_added():
    old = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
    new = _make_profile(["a", "b", "c"], {"a": "int64", "b": "object", "c": "float64"})
    diff = compute_schema_diff(old, new)
    assert diff["columns_added"] == ["c"]
    assert diff["columns_removed"] == []
    assert diff["type_changes"] == {}


def test_schema_diff_columns_removed():
    old = _make_profile(["a", "b", "c"], {"a": "int64", "b": "object", "c": "float64"})
    new = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
    diff = compute_schema_diff(old, new)
    assert diff["columns_added"] == []
    assert diff["columns_removed"] == ["c"]
    assert diff["type_changes"] == {}


def test_schema_diff_type_changes():
    old = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
    new = _make_profile(["a", "b"], {"a": "float64", "b": "object"})
    diff = compute_schema_diff(old, new)
    assert diff["columns_added"] == []
    assert diff["columns_removed"] == []
    assert "a" in diff["type_changes"]
    assert diff["type_changes"]["a"] == {"old": "int64", "new": "float64"}
    assert "b" not in diff["type_changes"]


def test_schema_diff_no_changes():
    old = _make_profile(["x", "y"], {"x": "int64", "y": "float64"})
    new = _make_profile(["x", "y"], {"x": "int64", "y": "float64"})
    diff = compute_schema_diff(old, new)
    assert diff["columns_added"] == []
    assert diff["columns_removed"] == []
    assert diff["type_changes"] == {}
    assert diff["row_count_old"] == diff["row_count_new"]


def test_schema_diff_row_count_changes():
    old = _make_profile(["a"], {"a": "int64"}, row_count=100)
    new = _make_profile(["a"], {"a": "int64"}, row_count=200)
    diff = compute_schema_diff(old, new)
    assert diff["row_count_old"] == 100
    assert diff["row_count_new"] == 200
    assert diff["columns_added"] == []
    assert diff["columns_removed"] == []
    assert diff["type_changes"] == {}
