"""Edge-case unit tests for pipeline_viz.graph_builder (Layer 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipeline_viz.graph_builder import _discover_notebooks, _merge_io, build_graph
from pipeline_viz.models import CodeEntry
from pipeline_viz.paths import normalize_rel


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


def _write_manifest(path: Path, codes: list[dict] | None = None) -> None:
    """Create a minimal pipeline-manifest.yaml."""
    manifest = {"project_root": ".", "codes": codes or []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(manifest, default_flow_style=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_empty_project(self, tmp_path: Path):
        """No notebooks, no manifest -> empty graph."""
        g = build_graph(tmp_path, None)
        assert g.nodes == []
        assert g.edges == []

    def test_project_with_only_manifest_no_notebooks(self, tmp_path: Path):
        """Manifest references notebooks that don't exist on disk."""
        manifest_path = tmp_path / "pipeline-manifest.yaml"
        _write_manifest(manifest_path, [
            {"path": "notebooks/step1.ipynb", "inputs": ["data/input.csv"], "outputs": ["data/output.csv"]},
        ])
        g = build_graph(tmp_path, manifest_path)
        # The notebook is added from manifest but parse_notebook_io won't find
        # code cells since the file doesn't exist, so only manifest I/O
        has_step1 = any("step1.ipynb" in n.path for n in g.nodes)
        assert has_step1

    def test_manifest_with_nonexistent_notebook_path(self, tmp_path: Path):
        """Manifest references a notebook that doesn't exist."""
        manifest_path = tmp_path / "pipeline-manifest.yaml"
        _write_manifest(manifest_path, [
            {"path": "nonexistent.ipynb", "inputs": [], "outputs": []},
        ])
        # Should not crash
        g = build_graph(tmp_path, manifest_path)
        assert any("nonexistent.ipynb" in n.path for n in g.nodes)

    def test_single_notebook_no_manifest(self, tmp_path: Path):
        """Project with one notebook, no manifest."""
        nb_path = tmp_path / "analysis.ipynb"
        _write_notebook(nb_path, ["import pandas as pd\npd.read_csv('data/input.csv')"])
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "input.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        g = build_graph(tmp_path, None)
        nb_nodes = [n for n in g.nodes if n.kind == "notebook"]
        assert len(nb_nodes) == 1
        assert nb_nodes[0].path == "analysis.ipynb"

    def test_runtime_io_merges(self, tmp_path: Path):
        """runtime_io adds additional I/O relationships."""
        nb_path = tmp_path / "nb.ipynb"
        _write_notebook(nb_path, ["x = 1"])
        runtime_io = {
            "nb.ipynb": {
                "inputs": ["data/runtime_in.csv"],
                "outputs": ["data/runtime_out.csv"],
            }
        }
        g = build_graph(tmp_path, None, runtime_io=runtime_io)
        data_paths = {n.path for n in g.nodes if n.kind == "data"}
        assert "data/runtime_in.csv" in data_paths
        assert "data/runtime_out.csv" in data_paths

    def test_notebook_in_skipped_dir_excluded(self, tmp_path: Path):
        """Notebooks in .venv, __pycache__, etc. are excluded."""
        venv_nb = tmp_path / ".venv" / "lib" / "test.ipynb"
        _write_notebook(venv_nb, ["x = 1"])
        regular_nb = tmp_path / "regular.ipynb"
        _write_notebook(regular_nb, ["y = 2"])
        g = build_graph(tmp_path, None)
        paths = [n.path for n in g.nodes if n.kind == "notebook"]
        assert "regular.ipynb" in paths
        assert all(".venv" not in p for p in paths)


# ---------------------------------------------------------------------------
# _merge_io
# ---------------------------------------------------------------------------

class TestMergeIo:
    def test_manifest_entries_merge_with_parsed(self, tmp_path: Path):
        entry = CodeEntry(
            path="nb.ipynb",
            inputs=["data/manifest_in.csv"],
            outputs=["data/manifest_out.csv"],
        )
        parsed_in = {"data/parsed_in.csv"}
        parsed_out = {"data/parsed_out.csv"}
        ins, outs = _merge_io(tmp_path, entry, parsed_in, parsed_out)
        assert "data/manifest_in.csv" in ins
        assert "data/parsed_in.csv" in ins
        assert "data/manifest_out.csv" in outs
        assert "data/parsed_out.csv" in outs

    def test_empty_normalize_rel_results_filtered(self, tmp_path: Path):
        entry = CodeEntry(
            path="nb.ipynb",
            inputs=["", "#comment"],
            outputs=["  "],
        )
        ins, outs = _merge_io(tmp_path, entry, set(), set())
        assert "" not in ins
        assert "" not in outs


# ---------------------------------------------------------------------------
# _discover_notebooks
# ---------------------------------------------------------------------------

class TestDiscoverNotebooks:
    def test_finds_notebooks(self, tmp_path: Path):
        _write_notebook(tmp_path / "a.ipynb")
        _write_notebook(tmp_path / "sub" / "b.ipynb")
        result = _discover_notebooks(tmp_path)
        assert "a.ipynb" in result
        assert "sub/b.ipynb" in result

    def test_skips_venv_git_pycache(self, tmp_path: Path):
        _write_notebook(tmp_path / ".venv" / "skip.ipynb")
        _write_notebook(tmp_path / ".git" / "skip.ipynb")
        _write_notebook(tmp_path / "__pycache__" / "skip.ipynb")
        _write_notebook(tmp_path / "keep.ipynb")
        result = _discover_notebooks(tmp_path)
        assert result == ["keep.ipynb"]

    def test_sorted_output(self, tmp_path: Path):
        _write_notebook(tmp_path / "z.ipynb")
        _write_notebook(tmp_path / "a.ipynb")
        _write_notebook(tmp_path / "m.ipynb")
        result = _discover_notebooks(tmp_path)
        assert result == sorted(result)

    def test_empty_project(self, tmp_path: Path):
        result = _discover_notebooks(tmp_path)
        assert result == []

    def test_skips_ipynb_checkpoints(self, tmp_path: Path):
        _write_notebook(tmp_path / ".ipynb_checkpoints" / "nb-checkpoint.ipynb")
        _write_notebook(tmp_path / "nb.ipynb")
        result = _discover_notebooks(tmp_path)
        assert result == ["nb.ipynb"]
