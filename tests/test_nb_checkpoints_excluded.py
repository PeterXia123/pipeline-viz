"""`.ipynb_checkpoints` 下的副本不应进入图。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_viz.graph_builder import build_graph


def _write_nb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": [{"cell_type": "code", "metadata": {}, "source": "# x\n", "outputs": []}],
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_notebooks_under_ipynb_checkpoints_not_in_graph(tmp_path: Path):
    _write_nb(tmp_path / "notebooks" / "real.ipynb")
    _write_nb(tmp_path / "notebooks" / ".ipynb_checkpoints" / "real-checkpoint.ipynb")

    man = tmp_path / "pipeline-manifest.yaml"
    man.write_text(
        'project_root: "."\ncodes:\n  - path: notebooks/real.ipynb\n    inputs: []\n    outputs: []\n',
        encoding="utf-8",
    )

    g = build_graph(tmp_path, man)
    ids = {n.id for n in g.nodes}
    assert "code:notebooks/real.ipynb" in ids
    assert not any("ipynb_checkpoints" in n.path for n in g.nodes)


def test_manifest_path_under_checkpoints_ignored(tmp_path: Path):
    """即使用 manifest 写了 checkpoint 路径，也不应进图。"""
    _write_nb(tmp_path / "notebooks" / "real.ipynb")
    _write_nb(tmp_path / "notebooks" / ".ipynb_checkpoints" / "real-checkpoint.ipynb")

    man = tmp_path / "pipeline-manifest.yaml"
    man.write_text(
        'project_root: "."\ncodes:\n'
        "  - path: notebooks/real.ipynb\n    inputs: []\n    outputs: []\n"
        "  - path: notebooks/.ipynb_checkpoints/real-checkpoint.ipynb\n"
        "    inputs: []\n    outputs: []\n",
        encoding="utf-8",
    )

    g = build_graph(tmp_path, man)
    assert len([n for n in g.nodes if n.kind == "notebook"]) == 1
    assert all("ipynb_checkpoints" not in n.path for n in g.nodes)
