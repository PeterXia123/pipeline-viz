"""Edge-case unit tests for pipeline_viz.py_imports (Layer 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_viz.py_imports import (
    _resolve_module,
    _to_rel_py,
    build_module_to_py_map,
    parse_notebook_py_imports_ordered,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_notebook(path: Path, code_cells: list[str] | None = None) -> None:
    """Create a minimal valid .ipynb file."""
    cells = []
    for src in (code_cells or []):
        cells.append({
            "cell_type": "code",
            "source": src,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb), encoding="utf-8")


def _write_py(path: Path, content: str = "") -> None:
    """Create a .py file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# build_module_to_py_map
# ---------------------------------------------------------------------------

class TestBuildModuleToPyMap:
    def test_finds_py_files(self, tmp_path: Path):
        _write_py(tmp_path / "utils.py", "x = 1")
        _write_py(tmp_path / "lib" / "helper.py", "y = 2")
        m = build_module_to_py_map(tmp_path)
        assert "utils" in m
        assert m["utils"] == "utils.py"
        assert "lib.helper" in m
        assert m["lib.helper"] == "lib/helper.py"

    def test_skips_venv_pycache(self, tmp_path: Path):
        _write_py(tmp_path / ".venv" / "pkg" / "mod.py", "x = 1")
        _write_py(tmp_path / "__pycache__" / "cached.py", "x = 1")
        _write_py(tmp_path / "real.py", "x = 1")
        m = build_module_to_py_map(tmp_path)
        assert "real" in m
        # .venv and __pycache__ modules should be excluded
        assert all(".venv" not in v for v in m.values())
        assert all("__pycache__" not in v for v in m.values())

    def test_shorter_path_preferred(self, tmp_path: Path):
        """If two .py files map to the same module name, prefer shorter path."""
        _write_py(tmp_path / "utils.py", "x = 1")
        _write_py(tmp_path / "a" / "b" / "c" / "utils.py", "x = 2")
        m = build_module_to_py_map(tmp_path)
        assert m["utils"] == "utils.py"

    def test_empty_project(self, tmp_path: Path):
        m = build_module_to_py_map(tmp_path)
        assert m == {}


# ---------------------------------------------------------------------------
# _resolve_module
# ---------------------------------------------------------------------------

class TestResolveModule:
    def test_exact_match(self):
        module_map = {"foo.bar": "foo/bar.py"}
        assert _resolve_module("foo.bar", module_map) == "foo/bar.py"

    def test_prefix_match(self):
        module_map = {"foo": "foo.py"}
        assert _resolve_module("foo.bar", module_map) == "foo.py"

    def test_no_match_returns_none(self):
        module_map = {"foo": "foo.py"}
        assert _resolve_module("bar", module_map) is None

    def test_longer_prefix_preferred(self):
        module_map = {"foo": "foo.py", "foo.bar": "foo/bar.py"}
        # Exact match takes priority
        assert _resolve_module("foo.bar", module_map) == "foo/bar.py"

    def test_deep_prefix_match(self):
        module_map = {"a.b.c": "a/b/c.py"}
        assert _resolve_module("a.b.c.d.e", module_map) == "a/b/c.py"


# ---------------------------------------------------------------------------
# parse_notebook_py_imports_ordered
# ---------------------------------------------------------------------------

class TestParseNotebookPyImportsOrdered:
    def test_import_statement(self, tmp_path: Path):
        _write_py(tmp_path / "mymodule.py", "x = 1")
        _write_notebook(tmp_path / "nb.ipynb", ["import mymodule"])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert "mymodule.py" in result

    def test_run_directive(self, tmp_path: Path):
        _write_py(tmp_path / "setup.py", "x = 1")
        _write_notebook(tmp_path / "nb.ipynb", ["%run setup.py"])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert "setup.py" in result

    def test_syntax_error_in_cell(self, tmp_path: Path):
        """A cell with a syntax error should not crash the parser."""
        _write_py(tmp_path / "good.py", "x = 1")
        _write_notebook(tmp_path / "nb.ipynb", [
            "import good",
            "def broken(\n  # syntax error",
            "y = 2",
        ])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert "good.py" in result

    def test_nonexistent_notebook_returns_empty(self, tmp_path: Path):
        module_map = {"mymod": "mymod.py"}
        result = parse_notebook_py_imports_ordered(tmp_path, "nosuch.ipynb", module_map)
        assert result == []

    def test_dedup_same_import_twice(self, tmp_path: Path):
        _write_py(tmp_path / "mymodule.py", "x = 1")
        _write_notebook(tmp_path / "nb.ipynb", [
            "import mymodule",
            "import mymodule",
        ])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert result.count("mymodule.py") == 1

    def test_from_import(self, tmp_path: Path):
        _write_py(tmp_path / "lib" / "utils.py", "def helper(): pass")
        _write_notebook(tmp_path / "nb.ipynb", ["from lib.utils import helper"])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert "lib/utils.py" in result

    def test_import_non_project_module_excluded(self, tmp_path: Path):
        """Imports like 'import pandas' should not appear in results."""
        _write_notebook(tmp_path / "nb.ipynb", ["import pandas as pd"])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert result == []

    def test_ordering_preserved(self, tmp_path: Path):
        """Imports should appear in the order they are encountered in cells."""
        _write_py(tmp_path / "alpha.py", "x = 1")
        _write_py(tmp_path / "beta.py", "y = 2")
        _write_notebook(tmp_path / "nb.ipynb", [
            "import beta",
            "import alpha",
        ])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert result.index("beta.py") < result.index("alpha.py")

    def test_run_with_quotes(self, tmp_path: Path):
        _write_py(tmp_path / "setup.py", "x = 1")
        _write_notebook(tmp_path / "nb.ipynb", ['%run "setup.py"'])
        module_map = build_module_to_py_map(tmp_path)
        result = parse_notebook_py_imports_ordered(tmp_path, "nb.ipynb", module_map)
        assert "setup.py" in result


# ---------------------------------------------------------------------------
# _to_rel_py
# ---------------------------------------------------------------------------

class TestToRelPy:
    def test_py_file_under_root(self, tmp_path: Path):
        _write_py(tmp_path / "script.py", "x = 1")
        result = _to_rel_py(tmp_path, "script.py")
        assert result == "script.py"

    def test_file_outside_root_returns_none(self, tmp_path: Path):
        result = _to_rel_py(tmp_path, "/some/other/place/script.py")
        assert result is None

    def test_non_py_file_returns_none(self, tmp_path: Path):
        result = _to_rel_py(tmp_path, "data.csv")
        assert result is None

    def test_quoted_path(self, tmp_path: Path):
        _write_py(tmp_path / "script.py", "x = 1")
        result = _to_rel_py(tmp_path, '"script.py"')
        assert result == "script.py"

    def test_single_quoted_path(self, tmp_path: Path):
        _write_py(tmp_path / "script.py", "x = 1")
        result = _to_rel_py(tmp_path, "'script.py'")
        assert result == "script.py"

    def test_subdirectory_py(self, tmp_path: Path):
        _write_py(tmp_path / "sub" / "mod.py", "x = 1")
        result = _to_rel_py(tmp_path, "sub/mod.py")
        assert result == "sub/mod.py"

    def test_absolute_path_under_root(self, tmp_path: Path):
        _write_py(tmp_path / "script.py", "x = 1")
        result = _to_rel_py(tmp_path, str(tmp_path / "script.py"))
        assert result == "script.py"

    def test_empty_string(self, tmp_path: Path):
        result = _to_rel_py(tmp_path, "")
        assert result is None

    def test_dotdot_escaping_root(self, tmp_path: Path):
        result = _to_rel_py(tmp_path, "../../../etc/passwd.py")
        assert result is None
