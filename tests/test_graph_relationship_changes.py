"""
当 notebook 对 .py 的 import 顺序 / 增减变化时，图中 notebook_call 边应随之变化。

本地查看「前后差异」示例：
  cd pipeline-viz && PYTHONPATH=src pytest tests/test_graph_relationship_changes.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.models import GraphPayload
from pipeline_viz.paths import code_id


def _write_minimal_ipynb(path: Path, source: str) -> None:
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


def _manifest(path: Path) -> Path:
    p = path / "pipeline-manifest.yaml"
    p.write_text(
        'project_root: "."\n'
        "codes:\n"
        "  - path: nb/chain.ipynb\n"
        "    inputs: []\n"
        "    outputs: []\n",
        encoding="utf-8",
    )
    return p


def _py_files(tmp: Path) -> None:
    (tmp / "lib").mkdir(parents=True, exist_ok=True)
    (tmp / "lib" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp / "lib" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tmp / "lib" / "c.py").write_text("c = 3\n", encoding="utf-8")


def _notebook_call_targets(g: GraphPayload, nb_rel: str) -> list[str]:
    nb_id = code_id(nb_rel)
    calls = [e for e in g.edges if e.kind == "notebook_call" and e.source == nb_id]
    id_to_path = {n.id: n.path for n in g.nodes}
    return [id_to_path.get(t, t) for t in (e.target for e in calls)]


def test_notebook_py_import_order_changes_notebook_call_order(tmp_path: Path):
    """同一单元格内交换两行 import，notebook_call 目标顺序应反转。"""
    _py_files(tmp_path)
    man = _manifest(tmp_path)
    nb_path = tmp_path / "nb" / "chain.ipynb"

    _write_minimal_ipynb(
        nb_path,
        "from lib.a import a\nfrom lib.b import b\n",
    )
    g1 = build_graph(tmp_path, man)
    order1 = _notebook_call_targets(g1, "nb/chain.ipynb")
    assert order1 == ["lib/a.py", "lib/b.py"]

    _write_minimal_ipynb(
        nb_path,
        "from lib.b import b\nfrom lib.a import a\n",
    )
    g2 = build_graph(tmp_path, man)
    order2 = _notebook_call_targets(g2, "nb/chain.ipynb")
    assert order2 == ["lib/b.py", "lib/a.py"]
    assert order1 != order2


def test_notebook_py_add_remove_changes_call_edges(tmp_path: Path):
    """增加 / 去掉对某个 .py 的 import，应多出或少了对应 notebook_call 边。"""
    _py_files(tmp_path)
    man = _manifest(tmp_path)
    nb_path = tmp_path / "nb" / "chain.ipynb"

    _write_minimal_ipynb(nb_path, "from lib.a import a\n")
    g1 = build_graph(tmp_path, man)
    assert _notebook_call_targets(g1, "nb/chain.ipynb") == ["lib/a.py"]

    _write_minimal_ipynb(nb_path, "from lib.a import a\nfrom lib.b import b\nfrom lib.c import c\n")
    g2 = build_graph(tmp_path, man)
    assert _notebook_call_targets(g2, "nb/chain.ipynb") == ["lib/a.py", "lib/b.py", "lib/c.py"]

    _write_minimal_ipynb(nb_path, "from lib.b import b\n")
    g3 = build_graph(tmp_path, man)
    assert _notebook_call_targets(g3, "nb/chain.ipynb") == ["lib/b.py"]


def test_demo_print_relationship_change(tmp_path: Path):
    """终端加 -s 可打印前后对比：`pytest ... -k demo_print -s`"""
    _py_files(tmp_path)
    man = _manifest(tmp_path)
    nb_path = tmp_path / "nb" / "chain.ipynb"

    _write_minimal_ipynb(nb_path, "from lib.a import a\nfrom lib.b import b\n")
    g_before = build_graph(tmp_path, man)
    _write_minimal_ipynb(nb_path, "from lib.b import b\nfrom lib.a import a\n")
    g_after = build_graph(tmp_path, man)

    print("\n--- notebook_call 顺序（交换 import 前）---")
    print(_notebook_call_targets(g_before, "nb/chain.ipynb"))
    print("--- notebook_call 顺序（交换 import 后）---")
    print(_notebook_call_targets(g_after, "nb/chain.ipynb"))
    eb = {(e.source, e.target, e.kind) for e in g_before.edges}
    ea = {(e.source, e.target, e.kind) for e in g_after.edges}
    print("--- (source,target,kind) 集合相同（仍是 notebook→a、notebook→b）---")
    print("same edge set:", eb == ea)
    print("说明：连接集合可不变，但边的列出顺序会变；布局里 py 节点上下顺序跟此顺序一致。")

    assert _notebook_call_targets(g_before, "nb/chain.ipynb") != _notebook_call_targets(
        g_after, "nb/chain.ipynb"
    )
    # 同一对 py 的 notebook_call 边仍存在，但顺序反映 import 顺序
