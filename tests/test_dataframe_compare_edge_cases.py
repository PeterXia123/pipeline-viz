"""Edge-case unit tests for pipeline_viz.dataframe_compare (Layer 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline_viz.dataframe_compare import (
    align_merge_key_types,
    compare_snapshot_blob_to_disk,
    is_dataframe_file,
    normalize_merge_keys,
    run_datacompy_compare,
)


# ---------------------------------------------------------------------------
# is_dataframe_file
# ---------------------------------------------------------------------------

class TestIsDataframeFile:
    def test_csv_true(self):
        assert is_dataframe_file("data.csv") is True

    def test_csv_uppercase_true(self):
        assert is_dataframe_file("DATA.CSV") is True

    def test_txt_false(self):
        assert is_dataframe_file("notes.txt") is False

    def test_parquet_true(self):
        assert is_dataframe_file("data.parquet") is True

    def test_pq_true(self):
        assert is_dataframe_file("data.pq") is True

    def test_pkl_true(self):
        assert is_dataframe_file("model.pkl") is True

    def test_pickle_true(self):
        assert is_dataframe_file("model.pickle") is True

    def test_tsv_true(self):
        assert is_dataframe_file("data.tsv") is True

    def test_py_false(self):
        assert is_dataframe_file("script.py") is False

    def test_json_false(self):
        assert is_dataframe_file("config.json") is False

    def test_no_extension_false(self):
        assert is_dataframe_file("noext") is False


# ---------------------------------------------------------------------------
# normalize_merge_keys
# ---------------------------------------------------------------------------

class TestNormalizeMergeKeys:
    def test_none_returns_empty(self):
        assert normalize_merge_keys(None) == []

    def test_empty_list_returns_empty(self):
        assert normalize_merge_keys([]) == []

    def test_whitespace_keys_filtered(self):
        assert normalize_merge_keys(["  ", "\t", ""]) == []

    def test_normal_keys_preserved(self):
        assert normalize_merge_keys(["id", "name"]) == ["id", "name"]

    def test_keys_stripped(self):
        assert normalize_merge_keys(["  id  ", " name "]) == ["id", "name"]

    def test_mixed_valid_and_whitespace(self):
        result = normalize_merge_keys(["id", "", "  ", "name"])
        assert result == ["id", "name"]


# ---------------------------------------------------------------------------
# align_merge_key_types
# ---------------------------------------------------------------------------

class TestAlignMergeKeyTypes:
    def test_same_types_no_change(self):
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        df2 = pd.DataFrame({"id": [4, 5, 6]})
        r1, r2 = align_merge_key_types(df1, df2, ["id"])
        assert r1["id"].dtype == df1["id"].dtype
        assert r2["id"].dtype == df2["id"].dtype

    def test_int_vs_float_numeric_coercion(self):
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        df2 = pd.DataFrame({"id": [1.0, 2.0, 3.0]})
        r1, r2 = align_merge_key_types(df1, df2, ["id"])
        # After coercion both should be numeric
        assert pd.api.types.is_numeric_dtype(r1["id"])
        assert pd.api.types.is_numeric_dtype(r2["id"])

    def test_numeric_vs_string_string_fallback(self):
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["a", "b", "c"]})
        r1, r2 = align_merge_key_types(df1, df2, ["id"])
        # Should fall back to string since "a", "b", "c" can't become numeric
        assert r1["id"].dtype == object
        assert r2["id"].dtype == object

    def test_key_not_in_columns_skipped(self):
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"name": ["a", "b"]})
        r1, r2 = align_merge_key_types(df1, df2, ["missing_key"])
        # DataFrames should be unchanged
        pd.testing.assert_frame_equal(r1, df1)
        pd.testing.assert_frame_equal(r2, df2)

    def test_does_not_modify_originals(self):
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"id": ["a", "b"]})
        original_dtype = df1["id"].dtype
        align_merge_key_types(df1, df2, ["id"])
        assert df1["id"].dtype == original_dtype


# ---------------------------------------------------------------------------
# run_datacompy_compare
# ---------------------------------------------------------------------------

class TestRunDatacompyCompare:
    def test_identical_csvs_match(self, tmp_path: Path):
        content = "id,value\n1,a\n2,b\n"
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text(content, encoding="utf-8")
        right.write_text(content, encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is True

    def test_different_csvs_no_match(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
        right.write_text("id,value\n1,a\n2,c\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is False

    def test_with_merge_keys(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
        right.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, ["id"])
        assert result["matches"] is True

    def test_missing_merge_key_column_error(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("id,value\n1,a\n", encoding="utf-8")
        right.write_text("id,value\n1,a\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, ["nonexistent_key"])
        assert "error" in result
        assert result["matches"] is False

    def test_no_merge_keys_row_order(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        right.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is True
        assert "_pipeline_row__" in result.get("join_columns", [])

    def test_empty_dataframes_zero_rows(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("a,b\n", encoding="utf-8")
        right.write_text("a,b\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is True

    def test_report_is_string(self, tmp_path: Path):
        content = "x\n1\n"
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text(content, encoding="utf-8")
        right.write_text(content, encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert isinstance(result.get("report", ""), str)


# ---------------------------------------------------------------------------
# compare_snapshot_blob_to_disk
# ---------------------------------------------------------------------------

class TestCompareSnapshotBlobToDisk:
    def test_basic_comparison(self, tmp_path: Path):
        blob = b"id,val\n1,a\n2,b\n"
        disk_file = tmp_path / "current.csv"
        disk_file.write_text("id,val\n1,a\n2,b\n", encoding="utf-8")
        result = compare_snapshot_blob_to_disk(blob, "data.csv", disk_file, None)
        assert result["matches"] is True

    def test_different_content(self, tmp_path: Path):
        blob = b"id,val\n1,a\n2,b\n"
        disk_file = tmp_path / "current.csv"
        disk_file.write_text("id,val\n1,a\n2,CHANGED\n", encoding="utf-8")
        result = compare_snapshot_blob_to_disk(blob, "data.csv", disk_file, None)
        assert result["matches"] is False

    def test_with_merge_keys(self, tmp_path: Path):
        blob = b"id,val\n1,a\n2,b\n"
        disk_file = tmp_path / "current.csv"
        disk_file.write_text("id,val\n1,a\n2,b\n", encoding="utf-8")
        result = compare_snapshot_blob_to_disk(blob, "data.csv", disk_file, ["id"])
        assert result["matches"] is True


# ---------------------------------------------------------------------------
# Edge case: empty DataFrames (0 rows)
# ---------------------------------------------------------------------------

class TestEmptyDataFrames:
    def test_compare_two_empty_csvs(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("col1,col2\n", encoding="utf-8")
        right.write_text("col1,col2\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is True

    def test_compare_empty_with_nonempty(self, tmp_path: Path):
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        left.write_text("col1,col2\n", encoding="utf-8")
        right.write_text("col1,col2\n1,2\n", encoding="utf-8")
        result = run_datacompy_compare(left, right, None)
        assert result["matches"] is False

    def test_align_empty_dataframes(self):
        df1 = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        df2 = pd.DataFrame({"id": pd.Series([], dtype="float64")})
        r1, r2 = align_merge_key_types(df1, df2, ["id"])
        assert len(r1) == 0
        assert len(r2) == 0
