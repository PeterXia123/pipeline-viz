"""Edge-case tests for pipeline_viz.paths module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline_viz.paths import (
    DATA_EXT,
    SCAN_SKIP_DIR_NAMES,
    code_id,
    data_id,
    is_under_root,
    normalize_rel,
    rel_has_scan_skip_dir,
)


# ---------------------------------------------------------------------------
# rel_has_scan_skip_dir
# ---------------------------------------------------------------------------

class TestRelHasScanSkipDir:
    def test_empty_string_returns_false(self):
        assert rel_has_scan_skip_dir("") is False

    def test_path_containing_dot_venv(self):
        assert rel_has_scan_skip_dir(".venv/lib/something.py") is True

    def test_path_containing_pycache(self):
        assert rel_has_scan_skip_dir("src/__pycache__/mod.cpython-312.pyc") is True

    def test_nested_skip_dir_deep(self):
        assert rel_has_scan_skip_dir("a/b/c/.git/config") is True

    def test_skip_dir_as_only_segment(self):
        assert rel_has_scan_skip_dir("__pycache__") is True

    def test_similar_but_not_matching_venv2(self):
        """'venv2' is not in the skip set, only 'venv'."""
        assert rel_has_scan_skip_dir("venv2/something.py") is False

    def test_similar_but_not_matching_dot_git_hooks(self):
        """.git-hooks is different from .git."""
        assert rel_has_scan_skip_dir(".git-hooks/pre-commit") is False

    def test_windows_backslashes_normalized(self):
        assert rel_has_scan_skip_dir("src\\.venv\\lib\\foo.py") is True

    def test_ipynb_checkpoints_skipped(self):
        assert rel_has_scan_skip_dir("notebooks/.ipynb_checkpoints/nb-checkpoint.ipynb") is True

    def test_node_modules_skipped(self):
        assert rel_has_scan_skip_dir("frontend/node_modules/package/index.js") is True

    def test_pipeline_viz_dir_skipped(self):
        assert rel_has_scan_skip_dir(".pipeline-viz/snapshots/abc.json") is True

    def test_plain_valid_path(self):
        assert rel_has_scan_skip_dir("src/pipeline_viz/paths.py") is False

    def test_skip_dir_in_filename_not_dir(self):
        """A file named '.venv' (not a directory segment) should still match
        because it's a segment of the path."""
        assert rel_has_scan_skip_dir(".venv") is True

    def test_all_skip_dirs_individually(self):
        for name in SCAN_SKIP_DIR_NAMES:
            assert rel_has_scan_skip_dir(f"root/{name}/file.py") is True

    def test_leading_trailing_slashes(self):
        assert rel_has_scan_skip_dir("/venv/foo") is True
        assert rel_has_scan_skip_dir("venv/foo/") is True


# ---------------------------------------------------------------------------
# normalize_rel
# ---------------------------------------------------------------------------

class TestNormalizeRel:
    def test_empty_string_returns_empty(self, tmp_path):
        assert normalize_rel(tmp_path, "") == ""

    def test_whitespace_only_returns_empty(self, tmp_path):
        assert normalize_rel(tmp_path, "   ") == ""

    def test_quoted_double(self, tmp_path):
        assert normalize_rel(tmp_path, '"src/foo.py"') == "src/foo.py"

    def test_quoted_single(self, tmp_path):
        assert normalize_rel(tmp_path, "'src/foo.py'") == "src/foo.py"

    def test_comment_starting_path(self, tmp_path):
        assert normalize_rel(tmp_path, "# this is a comment") == ""

    def test_leading_dot_slash_stripped(self, tmp_path):
        result = normalize_rel(tmp_path, "./src/foo.py")
        assert result == "src/foo.py"

    def test_dotdot_segments_stripped(self, tmp_path):
        result = normalize_rel(tmp_path, "src/../data/file.csv")
        assert result == "src/data/file.csv"

    def test_windows_backslashes(self, tmp_path):
        result = normalize_rel(tmp_path, "src\\pipeline_viz\\paths.py")
        assert result == "src/pipeline_viz/paths.py"

    def test_absolute_path_inside_root(self, tmp_path):
        abs_path = str(tmp_path / "src" / "foo.py")
        result = normalize_rel(tmp_path, abs_path)
        assert result == "src/foo.py"

    def test_absolute_path_outside_root(self, tmp_path):
        # An absolute path outside root produces a relative path (with ..)
        # but .. segments are stripped, so we get something
        result = normalize_rel(tmp_path, "/some/other/place/foo.py")
        assert isinstance(result, str)
        assert ".." not in result

    def test_plain_relative_path(self, tmp_path):
        assert normalize_rel(tmp_path, "data/raw.csv") == "data/raw.csv"


# ---------------------------------------------------------------------------
# is_under_root
# ---------------------------------------------------------------------------

class TestIsUnderRoot:
    def test_path_inside_root(self, tmp_path):
        child = tmp_path / "sub" / "file.py"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        assert is_under_root(tmp_path, child) is True

    def test_path_outside_root(self, tmp_path):
        other = tmp_path.parent / "outside"
        other.mkdir(exist_ok=True)
        assert is_under_root(tmp_path, other) is False

    def test_path_with_dotdot_resolves_inside(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        tricky = sub / ".." / "b"  # resolves to tmp_path/a/b
        assert is_under_root(tmp_path, tricky) is True

    def test_path_equal_to_root(self, tmp_path):
        assert is_under_root(tmp_path, tmp_path) is True

    def test_path_with_dotdot_resolves_outside(self, tmp_path):
        # tmp_path/../ is parent of tmp_path, so outside
        outside = tmp_path / ".." / ".."
        assert is_under_root(tmp_path, outside) is False


# ---------------------------------------------------------------------------
# data_id / code_id
# ---------------------------------------------------------------------------

class TestDataIdCodeId:
    def test_data_id_basic(self):
        assert data_id("data/raw.csv") == "data:data/raw.csv"

    def test_code_id_basic(self):
        assert code_id("src/pipeline.py") == "code:src/pipeline.py"

    def test_data_id_empty_string(self):
        assert data_id("") == "data:"

    def test_code_id_empty_string(self):
        assert code_id("") == "code:"

    def test_data_id_special_chars(self):
        assert data_id("data/my file (1).csv") == "data:data/my file (1).csv"

    def test_code_id_special_chars(self):
        assert code_id("src/my-module_v2.py") == "code:src/my-module_v2.py"


# ---------------------------------------------------------------------------
# DATA_EXT regex
# ---------------------------------------------------------------------------

class TestDataExtRegex:
    _ALL_EXTENSIONS = [
        ".csv", ".tsv", ".parquet", ".pq", ".json", ".jsonl", ".ndjson",
        ".feather", ".ftr", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5",
        ".txt", ".xml", ".yaml", ".yml", ".npz", ".npy", ".xlsx", ".xls",
        ".orc", ".ipc", ".arrow",
    ]

    @pytest.mark.parametrize("ext", _ALL_EXTENSIONS)
    def test_lowercase_extension_matches(self, ext):
        assert DATA_EXT.search(f"file{ext}") is not None

    @pytest.mark.parametrize("ext", _ALL_EXTENSIONS)
    def test_uppercase_extension_matches(self, ext):
        assert DATA_EXT.search(f"file{ext.upper()}") is not None

    @pytest.mark.parametrize("ext", _ALL_EXTENSIONS)
    def test_mixed_case_extension_matches(self, ext):
        mixed = ext[0:2] + ext[2:].upper()  # e.g., .cSV
        assert DATA_EXT.search(f"file{mixed}") is not None

    def test_double_extension_matches_last(self):
        assert DATA_EXT.search("archive.tar.csv") is not None

    def test_non_data_extension_no_match(self):
        assert DATA_EXT.search("script.py") is None

    def test_ipynb_no_match(self):
        assert DATA_EXT.search("notebook.ipynb") is None

    def test_extension_at_end_only(self):
        # ".csv" in the middle of a filename should not match
        assert DATA_EXT.search("file.csv.bak") is None

    def test_no_extension(self):
        assert DATA_EXT.search("Makefile") is None
