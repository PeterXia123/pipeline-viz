"""Edge-case tests for pipeline_viz.notebook_sanitize module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_viz.notebook_sanitize import (
    ipynb_code_fingerprint_bytes,
    ipynb_json_to_code_only_text,
    raw_file_sha256,
    sanitize_notebook_dict,
    stable_ipynb_sha256,
)


# ---------------------------------------------------------------------------
# ipynb_code_fingerprint_bytes
# ---------------------------------------------------------------------------

class TestIpynbCodeFingerprintBytes:
    def test_empty_cells_list(self):
        result = ipynb_code_fingerprint_bytes({"cells": []})
        assert result == b""

    def test_no_code_cells_only_markdown(self):
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": "# Title"},
                {"cell_type": "markdown", "source": "Some text"},
            ]
        }
        result = ipynb_code_fingerprint_bytes(nb)
        assert result == b""

    def test_source_as_string(self):
        nb = {
            "cells": [
                {"cell_type": "code", "source": "x = 1\n"},
            ]
        }
        result = ipynb_code_fingerprint_bytes(nb)
        assert b"x = 1" in result
        assert b"# --- code cell 0 ---" in result

    def test_source_as_list(self):
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["x = ", "1\n"]},
            ]
        }
        result = ipynb_code_fingerprint_bytes(nb)
        assert b"x = 1" in result

    def test_multiple_code_cells(self):
        nb = {
            "cells": [
                {"cell_type": "code", "source": "a = 1\n"},
                {"cell_type": "markdown", "source": "---"},
                {"cell_type": "code", "source": "b = 2\n"},
            ]
        }
        result = ipynb_code_fingerprint_bytes(nb)
        assert b"# --- code cell 0 ---" in result
        assert b"# --- code cell 1 ---" in result
        assert b"a = 1" in result
        assert b"b = 2" in result

    def test_non_dict_cells_skipped(self):
        nb = {
            "cells": [
                "not a dict",
                42,
                {"cell_type": "code", "source": "valid = True\n"},
            ]
        }
        result = ipynb_code_fingerprint_bytes(nb)
        assert b"valid = True" in result
        assert b"# --- code cell 0 ---" in result

    def test_no_cells_key(self):
        result = ipynb_code_fingerprint_bytes({})
        assert result == b""

    def test_cells_is_none(self):
        result = ipynb_code_fingerprint_bytes({"cells": None})
        assert result == b""


# ---------------------------------------------------------------------------
# ipynb_json_to_code_only_text
# ---------------------------------------------------------------------------

class TestIpynbJsonToCodeOnlyText:
    def test_empty_string_returns_empty(self):
        assert ipynb_json_to_code_only_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert ipynb_json_to_code_only_text("   \n\t  ") == ""

    def test_invalid_json_returns_original(self):
        bad = "{not valid json"
        assert ipynb_json_to_code_only_text(bad) == bad

    def test_valid_non_dict_json_returns_original(self):
        arr = json.dumps([1, 2, 3])
        assert ipynb_json_to_code_only_text(arr) == arr

    def test_valid_string_json_returns_original(self):
        s = json.dumps("just a string")
        assert ipynb_json_to_code_only_text(s) == s

    def test_valid_notebook_json(self):
        nb = json.dumps({
            "cells": [
                {"cell_type": "code", "source": "print('hi')\n"},
                {"cell_type": "markdown", "source": "# ignored"},
            ]
        })
        result = ipynb_json_to_code_only_text(nb)
        assert "print('hi')" in result
        assert "# ignored" not in result

    def test_notebook_with_no_code_cells(self):
        nb = json.dumps({
            "cells": [
                {"cell_type": "markdown", "source": "no code here"},
            ]
        })
        result = ipynb_json_to_code_only_text(nb)
        assert result == ""


# ---------------------------------------------------------------------------
# sanitize_notebook_dict
# ---------------------------------------------------------------------------

class TestSanitizeNotebookDict:
    def test_removes_execution_count(self):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x = 1\n",
                    "execution_count": 42,
                    "metadata": {},
                    "outputs": [],
                },
            ],
            "metadata": {},
        }
        result = sanitize_notebook_dict(nb)
        assert result["cells"][0]["execution_count"] is None

    def test_removes_metadata_execution(self):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x = 1\n",
                    "execution_count": 1,
                    "metadata": {"execution": {"started": "2024-01-01T00:00:00Z"}},
                    "outputs": [],
                },
            ],
            "metadata": {"execution": {"kernel": "python3"}},
        }
        result = sanitize_notebook_dict(nb)
        assert "execution" not in result["metadata"]
        assert "execution" not in result["cells"][0]["metadata"]

    def test_removes_output_metadata(self):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "print(1)\n",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [
                        {"output_type": "stream", "text": "1\n", "metadata": {"isolated": True}},
                    ],
                },
            ],
            "metadata": {},
        }
        result = sanitize_notebook_dict(nb)
        assert "metadata" not in result["cells"][0]["outputs"][0]

    def test_handles_missing_metadata_key(self):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x = 1\n",
                    "execution_count": 5,
                    # no "metadata" key
                    "outputs": [],
                },
            ],
            # no top-level "metadata" key
        }
        result = sanitize_notebook_dict(nb)
        assert result["cells"][0]["execution_count"] is None

    def test_handles_non_dict_cells(self):
        nb = {
            "cells": ["not a dict", 123],
            "metadata": {},
        }
        # Should not raise
        result = sanitize_notebook_dict(nb)
        assert result["cells"] == ["not a dict", 123]

    def test_markdown_cells_untouched(self):
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": "# Hello", "metadata": {}},
            ],
            "metadata": {},
        }
        result = sanitize_notebook_dict(nb)
        assert result["cells"][0]["source"] == "# Hello"
        # execution_count not set on markdown cells because cell_type != "code"

    def test_no_cells_key_no_crash(self):
        nb = {"metadata": {}}
        result = sanitize_notebook_dict(nb)
        assert result == {"metadata": {}}

    def test_empty_outputs_list(self):
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "",
                    "execution_count": 0,
                    "metadata": {},
                    "outputs": [],
                },
            ],
            "metadata": {},
        }
        result = sanitize_notebook_dict(nb)
        assert result["cells"][0]["outputs"] == []


# ---------------------------------------------------------------------------
# stable_ipynb_sha256
# ---------------------------------------------------------------------------

class TestStableIpynbSha256:
    def test_valid_notebook(self, tmp_path):
        nb = {
            "cells": [
                {"cell_type": "code", "source": "x = 1\n", "metadata": {}, "outputs": []},
            ],
            "metadata": {},
            "nbformat": 4,
        }
        f = tmp_path / "nb.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        result = stable_ipynb_sha256(f)
        assert len(result) == 64  # SHA-256 hex length
        # Verify it matches manual computation
        expected_body = ipynb_code_fingerprint_bytes(nb)
        expected = hashlib.sha256(expected_body).hexdigest()
        assert result == expected

    def test_notebook_with_no_code_cells(self, tmp_path):
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": "# Just text"},
            ],
            "metadata": {},
        }
        f = tmp_path / "md_only.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        result = stable_ipynb_sha256(f)
        # Hash of empty code fingerprint
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_same_code_different_outputs_same_hash(self, tmp_path):
        code = "x = 1\n"
        nb1 = {"cells": [{"cell_type": "code", "source": code, "outputs": []}]}
        nb2 = {"cells": [{"cell_type": "code", "source": code, "outputs": [{"text": "1"}]}]}
        f1 = tmp_path / "nb1.ipynb"
        f2 = tmp_path / "nb2.ipynb"
        f1.write_text(json.dumps(nb1), encoding="utf-8")
        f2.write_text(json.dumps(nb2), encoding="utf-8")
        assert stable_ipynb_sha256(f1) == stable_ipynb_sha256(f2)

    def test_invalid_json_falls_back_to_raw_hash(self, tmp_path):
        f = tmp_path / "bad.ipynb"
        content = "this is not json"
        f.write_text(content, encoding="utf-8")
        result = stable_ipynb_sha256(f)
        # Should fall back to raw file hash
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert result == expected

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = stable_ipynb_sha256(tmp_path / "missing.ipynb")
        assert result == ""

    def test_valid_json_but_not_dict_falls_back(self, tmp_path):
        f = tmp_path / "array.ipynb"
        f.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = stable_ipynb_sha256(f)
        # Not a dict -> falls back to raw hash
        expected = raw_file_sha256(f)
        assert result == expected


# ---------------------------------------------------------------------------
# raw_file_sha256
# ---------------------------------------------------------------------------

class TestRawFileSha256:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "data.txt"
        content = b"hello world\n"
        f.write_bytes(content)
        result = raw_file_sha256(f)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_nonexistent_file_returns_empty(self, tmp_path):
        assert raw_file_sha256(tmp_path / "nope.bin") == ""

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = raw_file_sha256(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_binary_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        data = bytes(range(256))
        f.write_bytes(data)
        result = raw_file_sha256(f)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected
