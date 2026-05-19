"""
1) .py 多输入/多输出
2) 一个 .py 的 input 是另一个 .py 的 output（经 data 文件串联）
3) 仅 notebook 引用到的 .py 入图；目录中其它 .py 及仅 manifest 声明但未 import 的 .py 不入图
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id

FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def root_multi():
    return FIX / "py_multi_io"


@pytest.fixture
def root_chain():
    return FIX / "py_chain"


@pytest.fixture
def root_unused():
    return FIX / "py_unused"


def test_py_file_multiple_inputs_outputs(root_multi):
    g = build_graph(root_multi, root_multi / "pipeline-manifest.yaml")
    py = "pipeline/multi_stage.py"
    pid = code_id(py)

    ins = {e.source for e in g.edges if e.target == pid and e.kind == "input"}
    outs = {e.target for e in g.edges if e.source == pid and e.kind == "output"}

    assert ins == {data_id("data/in_a.csv"), data_id("data/in_b.csv")}
    assert outs == {data_id("data/out_1.csv"), data_id("data/out_2.csv")}

    paths = {n.path for n in g.nodes if n.kind == "python"}
    assert paths == {py}


def test_py_output_feeds_another_py_input(root_chain):
    g = build_graph(root_chain, root_chain / "pipeline-manifest.yaml")
    es = {(x.source, x.target, x.kind) for x in g.edges}

    assert (data_id("data/raw.csv"), code_id("scripts/step1.py"), "input") in es
    assert (code_id("scripts/step1.py"), data_id("data/mid.csv"), "output") in es
    assert (data_id("data/mid.csv"), code_id("scripts/step2.py"), "input") in es
    assert (code_id("scripts/step2.py"), data_id("data/final.csv"), "output") in es

    py_paths = sorted(n.path for n in g.nodes if n.kind == "python")
    assert py_paths == ["scripts/step1.py", "scripts/step2.py"]


def test_unused_py_on_disk_and_manifest_only_not_shown(root_unused):
    g = build_graph(root_unused, root_unused / "pipeline-manifest.yaml")
    paths = {n.path for n in g.nodes}

    assert "lib/orphan.py" not in paths
    assert "data/ghost.csv" not in paths
    assert "data/ghost_out.csv" not in paths
    assert "lib/used.py" in paths

    assert not any("orphan" in p for p in paths)
