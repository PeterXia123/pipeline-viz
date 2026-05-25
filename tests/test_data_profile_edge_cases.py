"""Edge-case unit tests for pipeline_viz.data_profile (Layer 2)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

from pipeline_viz.data_profile import (
    compute_schema_diff,
    extract_profile,
    extract_profile_from_bytes,
    is_dataframe_format,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_profile(
    columns: list[str] | None = None,
    dtypes: dict[str, str] | None = None,
    row_count: int | None = 10,
    col_count: int | None = None,
    file_size: int = 100,
) -> dict[str, Any]:
    cols = columns or []
    return {
        "row_count": row_count,
        "col_count": col_count if col_count is not None else len(cols),
        "columns": cols,
        "dtypes": dtypes or {},
        "file_size": file_size,
        "format": "csv",
        "error": None,
    }


# ---------------------------------------------------------------------------
# is_dataframe_format
# ---------------------------------------------------------------------------

class TestIsDataframeFormat:
    def test_csv_true(self):
        assert is_dataframe_format("data.csv") is True

    def test_csv_uppercase_true(self):
        assert is_dataframe_format("DATA.CSV") is True

    def test_py_false(self):
        assert is_dataframe_format("script.py") is False

    def test_json_false(self):
        assert is_dataframe_format("config.json") is False

    def test_parquet_true(self):
        assert is_dataframe_format("data.parquet") is True

    def test_pq_true(self):
        assert is_dataframe_format("data.pq") is True

    def test_pkl_true(self):
        assert is_dataframe_format("model.pkl") is True

    def test_pickle_true(self):
        assert is_dataframe_format("model.pickle") is True

    def test_tsv_true(self):
        assert is_dataframe_format("data.tsv") is True

    def test_yaml_false(self):
        assert is_dataframe_format("config.yaml") is False

    def test_no_extension_false(self):
        assert is_dataframe_format("noextension") is False

    def test_h5_false(self):
        assert is_dataframe_format("model.h5") is False

    def test_mixed_case_parquet(self):
        assert is_dataframe_format("data.Parquet") is True


# ---------------------------------------------------------------------------
# extract_profile
# ---------------------------------------------------------------------------

class TestExtractProfile:
    def test_missing_file(self, tmp_path: Path):
        p = extract_profile(tmp_path / "nope.csv")
        assert p["format"] == "missing"
        assert p["row_count"] is None
        assert p["file_size"] == 0
        assert p["error"] is not None

    def test_non_dataframe_extension(self, tmp_path: Path):
        f = tmp_path / "script.py"
        f.write_text("print('hello')\n", encoding="utf-8")
        p = extract_profile(f)
        assert p["format"] == "non_dataframe"
        assert p["row_count"] is None
        assert p["file_size"] > 0
        assert p["error"] is None

    def test_valid_csv(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
        p = extract_profile(f)
        assert p["row_count"] == 2
        assert p["col_count"] == 3
        assert p["columns"] == ["a", "b", "c"]
        assert p["format"] == "csv"
        assert p["error"] is None

    def test_empty_csv_headers_only(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("a,b,c\n", encoding="utf-8")
        p = extract_profile(f)
        assert p["row_count"] == 0
        assert p["col_count"] == 3
        assert p["columns"] == ["a", "b", "c"]

    def test_csv_no_headers(self, tmp_path: Path):
        # A CSV with no header row: pandas treats first row as header
        f = tmp_path / "noheader.csv"
        f.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
        p = extract_profile(f)
        # pandas reads first row as headers
        assert p["row_count"] == 1
        assert p["col_count"] == 3

    def test_pkl_not_dataframe(self, tmp_path: Path):
        f = tmp_path / "model.pkl"
        # pickle a plain dict, not a DataFrame
        f.write_bytes(pickle.dumps({"key": "value"}))
        p = extract_profile(f)
        assert p["format"] == "non_dataframe"
        assert p["row_count"] is None

    def test_pkl_valid_dataframe(self, tmp_path: Path):
        import pandas as pd

        f = tmp_path / "df.pkl"
        pd.DataFrame({"x": [1, 2, 3]}).to_pickle(f)
        p = extract_profile(f)
        assert p["row_count"] == 3
        assert p["col_count"] == 1
        assert p["columns"] == ["x"]

    def test_corrupted_csv(self, tmp_path: Path):
        # Binary garbage with .csv extension
        f = tmp_path / "bad.csv"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        # pandas may or may not raise -- either way we should get a result
        p = extract_profile(f)
        assert "format" in p


# ---------------------------------------------------------------------------
# extract_profile_from_bytes
# ---------------------------------------------------------------------------

class TestExtractProfileFromBytes:
    def test_valid_csv_bytes(self):
        data = b"x,y\n1,2\n3,4\n"
        p = extract_profile_from_bytes(data, ".csv")
        assert p["row_count"] == 2
        assert p["col_count"] == 2
        assert p["columns"] == ["x", "y"]
        assert p["error"] is None

    def test_corrupted_bytes(self):
        data = b"\x00\x01\x02\xff\xfe\xfd"
        p = extract_profile_from_bytes(data, ".parquet")
        assert p["error"] is not None
        assert p["row_count"] is None

    def test_empty_csv_bytes(self):
        data = b"a,b\n"
        p = extract_profile_from_bytes(data, ".csv")
        assert p["row_count"] == 0
        assert p["col_count"] == 2

    def test_tsv_bytes(self):
        data = b"a\tb\n1\t2\n"
        p = extract_profile_from_bytes(data, ".tsv")
        assert p["row_count"] == 1
        assert p["col_count"] == 2


# ---------------------------------------------------------------------------
# compute_schema_diff
# ---------------------------------------------------------------------------

class TestComputeSchemaDiff:
    def test_no_changes(self):
        old = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
        new = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
        diff = compute_schema_diff(old, new)
        assert diff["columns_added"] == []
        assert diff["columns_removed"] == []
        assert diff["type_changes"] == {}

    def test_columns_added(self):
        old = _make_profile(["a"], {"a": "int64"})
        new = _make_profile(["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
        diff = compute_schema_diff(old, new)
        assert sorted(diff["columns_added"]) == ["b", "c"]
        assert diff["columns_removed"] == []

    def test_columns_removed(self):
        old = _make_profile(["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
        new = _make_profile(["a"], {"a": "int64"})
        diff = compute_schema_diff(old, new)
        assert diff["columns_added"] == []
        assert sorted(diff["columns_removed"]) == ["b", "c"]

    def test_type_changes(self):
        old = _make_profile(["a", "b"], {"a": "int64", "b": "object"})
        new = _make_profile(["a", "b"], {"a": "float64", "b": "object"})
        diff = compute_schema_diff(old, new)
        assert "a" in diff["type_changes"]
        assert diff["type_changes"]["a"]["old"] == "int64"
        assert diff["type_changes"]["a"]["new"] == "float64"
        assert "b" not in diff["type_changes"]

    def test_both_empty_profiles(self):
        old = _make_profile([], {})
        new = _make_profile([], {})
        diff = compute_schema_diff(old, new)
        assert diff["columns_added"] == []
        assert diff["columns_removed"] == []
        assert diff["type_changes"] == {}

    def test_none_columns_list(self):
        old = {"columns": None, "dtypes": None, "row_count": None, "col_count": None, "file_size": 0}
        new = {"columns": None, "dtypes": None, "row_count": None, "col_count": None, "file_size": 0}
        diff = compute_schema_diff(old, new)
        assert diff["columns_added"] == []
        assert diff["columns_removed"] == []
        assert diff["type_changes"] == {}

    def test_mixed_add_remove_type_change(self):
        old = _make_profile(["a", "b", "c"], {"a": "int64", "b": "object", "c": "float64"})
        new = _make_profile(["a", "c", "d"], {"a": "float64", "c": "float64", "d": "int64"})
        diff = compute_schema_diff(old, new)
        assert diff["columns_added"] == ["d"]
        assert diff["columns_removed"] == ["b"]
        assert "a" in diff["type_changes"]
        assert diff["type_changes"]["a"]["old"] == "int64"
        assert diff["type_changes"]["a"]["new"] == "float64"

    def test_row_count_tracked(self):
        old = _make_profile(["a"], {"a": "int64"}, row_count=100)
        new = _make_profile(["a"], {"a": "int64"}, row_count=200)
        diff = compute_schema_diff(old, new)
        assert diff["row_count_old"] == 100
        assert diff["row_count_new"] == 200

    def test_file_size_tracked(self):
        old = _make_profile([], {}, file_size=500)
        new = _make_profile([], {}, file_size=1000)
        diff = compute_schema_diff(old, new)
        assert diff["file_size_old"] == 500
        assert diff["file_size_new"] == 1000
