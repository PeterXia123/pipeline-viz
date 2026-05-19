"""
新增读取 data/product.csv 时，图上应出现新的 data 节点与 input 边。

两种常见入口：
1) notebook 源码里新增 pd.read_csv("data/product.csv")（解析器扫描）；
2) 仅改 manifest 的 inputs 列表（图仍从 manifest 构图，但 manifest 文件本身不进入图、也不参与快照跟踪）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id


def _write_nb(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_notebook_new_read_csv_product_shows_new_edge(tmp_path: Path):
    """pipeline notebook 新 read product.csv → 多一条 data→notebook 的 input 边。"""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "in.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "data" / "product.csv").write_text("x\ny\n", encoding="utf-8")
    nb_rel = "notebooks/run.ipynb"
    _write_nb(
        tmp_path / nb_rel,
        'import pandas as pd\npd.read_csv("data/in.csv")\n',
    )
    man = tmp_path / "pipeline-manifest.yaml"
    man.write_text(
        f'project_root: "."\n'
        f"codes:\n"
        f"  - path: {nb_rel}\n"
        f"    inputs: []\n"
        f"    outputs: []\n",
        encoding="utf-8",
    )

    g1 = build_graph(tmp_path, man)
    nb_id = code_id(nb_rel)
    es1 = {(e.source, e.target, e.kind) for e in g1.edges}
    assert (data_id("data/in.csv"), nb_id, "input") in es1
    assert (data_id("data/product.csv"), nb_id, "input") not in es1

    _write_nb(
        tmp_path / nb_rel,
        'import pandas as pd\npd.read_csv("data/in.csv")\npd.read_csv("data/product.csv")\n',
    )
    g2 = build_graph(tmp_path, man)
    es2 = {(e.source, e.target, e.kind) for e in g2.edges}
    assert (data_id("data/in.csv"), nb_id, "input") in es2
    assert (data_id("data/product.csv"), nb_id, "input") in es2
    assert data_id("data/product.csv") in {n.id for n in g2.nodes}
    assert len(g2.nodes) == len(g1.nodes) + 1


def test_manifest_new_inputs_product_shows_new_edge(tmp_path: Path):
    """不改 notebook，只在 manifest 里为 notebook 增加 input data/product.csv → 新边。"""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "base.csv").write_text("k\n", encoding="utf-8")
    (tmp_path / "data" / "product.csv").write_text("k\n", encoding="utf-8")
    nb_rel = "notebooks/run.ipynb"
    _write_nb(tmp_path / nb_rel, "# no dynamic io\n")
    man = tmp_path / "pipeline-manifest.yaml"

    man.write_text(
        f'project_root: "."\n'
        f"codes:\n"
        f"  - path: {nb_rel}\n"
        f"    inputs:\n"
        f"      - data/base.csv\n"
        f"    outputs: []\n",
        encoding="utf-8",
    )
    g1 = build_graph(tmp_path, man)
    nb_id = code_id(nb_rel)
    es1 = {(e.source, e.target, e.kind) for e in g1.edges}
    assert (data_id("data/base.csv"), nb_id, "input") in es1
    assert (data_id("data/product.csv"), nb_id, "input") not in es1

    man.write_text(
        f'project_root: "."\n'
        f"codes:\n"
        f"  - path: {nb_rel}\n"
        f"    inputs:\n"
        f"      - data/base.csv\n"
        f"      - data/product.csv\n"
        f"    outputs: []\n",
        encoding="utf-8",
    )
    g2 = build_graph(tmp_path, man)
    es2 = {(e.source, e.target, e.kind) for e in g2.edges}
    assert (data_id("data/product.csv"), nb_id, "input") in es2
    assert data_id("data/product.csv") in {n.id for n in g2.nodes}
