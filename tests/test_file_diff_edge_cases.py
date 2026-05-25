"""Edge-case tests for pipeline_viz.file_diff module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_viz.file_diff import (
    changed_lines_from_unified,
    diff_pair_for_display,
    read_text_safe,
    unified_diff_text,
)


# ---------------------------------------------------------------------------
# read_text_safe
# ---------------------------------------------------------------------------

class TestReadTextSafe:
    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert read_text_safe(tmp_path / "nope.txt") == ""

    def test_file_exactly_at_max_bytes(self, tmp_path):
        f = tmp_path / "exact.txt"
        content = "A" * 100
        f.write_text(content, encoding="utf-8")
        result = read_text_safe(f, max_bytes=100)
        assert result == content

    def test_file_over_max_bytes_truncation_message(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("B" * 200, encoding="utf-8")
        result = read_text_safe(f, max_bytes=100)
        assert "100 bytes" in result
        assert "省略" in result or "过大" in result

    def test_file_under_max_bytes(self, tmp_path):
        f = tmp_path / "small.txt"
        content = "Hello world\n"
        f.write_text(content, encoding="utf-8")
        result = read_text_safe(f, max_bytes=400_000)
        assert result == content

    def test_file_with_encoding_errors(self, tmp_path):
        f = tmp_path / "binary.txt"
        f.write_bytes(b"Hello \xff\xfe World\n")
        result = read_text_safe(f)
        # errors="replace" should produce the replacement character
        assert "Hello" in result
        assert "World" in result

    def test_directory_returns_empty(self, tmp_path):
        """A directory is not a file, so is_file() returns False."""
        assert read_text_safe(tmp_path) == ""


# ---------------------------------------------------------------------------
# unified_diff_text
# ---------------------------------------------------------------------------

class TestUnifiedDiffText:
    def test_identical_strings_empty_diff(self):
        assert unified_diff_text("file.py", "hello\n", "hello\n") == ""

    def test_one_empty_before(self):
        result = unified_diff_text("file.py", "", "line1\n")
        assert "+line1" in result

    def test_one_empty_after(self):
        result = unified_diff_text("file.py", "line1\n", "")
        assert "-line1" in result

    def test_both_empty_strings(self):
        assert unified_diff_text("file.py", "", "") == ""

    def test_crlf_line_endings(self):
        before = "line1\r\nline2\r\n"
        after = "line1\r\nmodified\r\n"
        result = unified_diff_text("file.py", before, after)
        assert "-line2" in result
        assert "+modified" in result

    def test_diff_header_contains_path(self):
        result = unified_diff_text("src/foo.py", "old\n", "new\n")
        assert "a/src/foo.py" in result
        assert "b/src/foo.py" in result


# ---------------------------------------------------------------------------
# diff_pair_for_display
# ---------------------------------------------------------------------------

class TestDiffPairForDisplay:
    def test_regular_text_file(self):
        result = diff_pair_for_display("script.py", "a = 1\n", "a = 2\n")
        assert result["path"] == "script.py"
        assert result["unchanged"] is False
        assert result["diff_kind"] == "text"
        assert result["unified"]  # non-empty diff

    def test_identical_files_unchanged_true(self):
        text = "x = 42\n"
        result = diff_pair_for_display("script.py", text, text)
        assert result["unchanged"] is True
        assert result["unified"] == ""

    def test_empty_files(self):
        result = diff_pair_for_display("empty.txt", "", "")
        assert result["unchanged"] is True

    def test_ipynb_triggers_code_only_extraction(self):
        nb_before = json.dumps({
            "cells": [
                {"cell_type": "code", "source": "x = 1\n", "metadata": {}, "outputs": []},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        })
        nb_after = json.dumps({
            "cells": [
                {"cell_type": "code", "source": "x = 2\n", "metadata": {}, "outputs": []},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        })
        result = diff_pair_for_display("notebook.ipynb", nb_before, nb_after)
        assert result["diff_kind"] == "ipynb_code_only"
        assert result["unchanged"] is False

    def test_ipynb_same_code_different_outputs_unchanged(self):
        """If code cells are identical but outputs differ, should be unchanged."""
        nb1 = json.dumps({
            "cells": [
                {"cell_type": "code", "source": "print(1)\n",
                 "metadata": {}, "outputs": [{"text": "1\n"}]},
            ],
            "metadata": {},
            "nbformat": 4,
        })
        nb2 = json.dumps({
            "cells": [
                {"cell_type": "code", "source": "print(1)\n",
                 "metadata": {}, "outputs": [{"text": "1\n", "extra": True}]},
            ],
            "metadata": {},
            "nbformat": 4,
        })
        result = diff_pair_for_display("nb.ipynb", nb1, nb2)
        assert result["unchanged"] is True

    def test_ipynb_uppercase_extension(self):
        nb = json.dumps({"cells": [], "metadata": {}, "nbformat": 4})
        result = diff_pair_for_display("NB.IPYNB", nb, nb)
        assert result["diff_kind"] == "ipynb_code_only"


# ---------------------------------------------------------------------------
# changed_lines_from_unified
# ---------------------------------------------------------------------------

class TestChangedLinesFromUnified:
    def test_empty_string_returns_empty_lists(self):
        result = changed_lines_from_unified("")
        assert result == {"old": [], "new": []}

    def test_simple_addition(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,3 @@\n"
            " line1\n"
            "+inserted\n"
            " line2\n"
        )
        result = changed_lines_from_unified(diff)
        assert result["new"] == [2]
        assert result["old"] == []

    def test_simple_deletion(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,2 @@\n"
            " line1\n"
            "-removed\n"
            " line3\n"
        )
        result = changed_lines_from_unified(diff)
        assert result["old"] == [2]
        assert result["new"] == []

    def test_modification(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-old_line\n"
            "+new_line\n"
            " line3\n"
        )
        result = changed_lines_from_unified(diff)
        assert result["old"] == [2]
        assert result["new"] == [2]

    def test_multiple_hunks(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " c\n"
            "@@ -10,3 +10,3 @@\n"
            " x\n"
            "-y\n"
            "+Y\n"
            " z\n"
        )
        result = changed_lines_from_unified(diff)
        assert result["old"] == [2, 11]
        assert result["new"] == [2, 11]
