from __future__ import annotations

from pathlib import Path
from typing import Literal

from pipeline_viz.manifest_loader import load_manifest, manifest_by_notebook
from pipeline_viz.models import CodeEntry, GraphEdge, GraphNode, GraphPayload
from pipeline_viz.notebook_parser import parse_notebook_io, parse_python_file_io
from pipeline_viz.paths import (
    SCAN_SKIP_DIR_NAMES,
    code_id,
    data_id,
    normalize_rel,
    rel_has_scan_skip_dir,
)
from pipeline_viz.py_imports import build_module_to_py_map, parse_notebook_py_imports_ordered


def _merge_io(
    root: Path,
    entry: CodeEntry,
    parsed_in: set[str],
    parsed_out: set[str],
) -> tuple[set[str], set[str]]:
    ins = {normalize_rel(root, x) for x in entry.inputs} | parsed_in
    outs = {normalize_rel(root, x) for x in entry.outputs} | parsed_out
    return {x for x in ins if x}, {x for x in outs if x}


def _discover_notebooks(project_root: Path) -> list[str]:
    out: list[str] = []
    for p in project_root.rglob("*.ipynb"):
        if any(part in SCAN_SKIP_DIR_NAMES for part in p.parts):
            continue
        rel = p.relative_to(project_root.resolve()).as_posix()
        out.append(rel)
    return sorted(out)


def build_graph(
    project_root: Path,
    manifest_path: Path | None,
    runtime_io: dict[str, dict[str, list[str]]] | None = None,
) -> GraphPayload:
    root = project_root.resolve()
    manifest_codes: dict[str, CodeEntry] = {}
    if manifest_path and manifest_path.is_file():
        m = load_manifest(manifest_path)
        manifest_codes = manifest_by_notebook(m)

    module_map = build_module_to_py_map(root)

    notebooks = _discover_notebooks(root)
    for rel in list(manifest_codes.keys()):
        if rel not in notebooks and rel.endswith(".ipynb"):
            if rel_has_scan_skip_dir(rel):
                continue
            notebooks.append(rel)
    notebooks = sorted(set(notebooks))

    # 仅被 notebook import / %run 引用到的 .py 会进入图；目录中其它 .py 不创建节点
    py_used: set[str] = set()
    for nb_rel in notebooks:
        for py_rel in parse_notebook_py_imports_ordered(root, nb_rel, module_map):
            py_used.add(py_rel)

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_i = 0
    io_seen: set[tuple[str, str, str]] = set()

    def ensure_data(rel: str) -> None:
        rid = data_id(rel)
        if rid not in nodes:
            label = Path(rel).name
            nodes[rid] = GraphNode(id=rid, label=label, kind="data", path=rel)

    def ensure_notebook(rel: str) -> None:
        cid = code_id(rel)
        if cid not in nodes:
            label = Path(rel).name
            nodes[cid] = GraphNode(id=cid, label=label, kind="notebook", path=rel)

    def ensure_python(rel: str) -> None:
        cid = code_id(rel)
        if cid not in nodes:
            label = Path(rel).name
            nodes[cid] = GraphNode(id=cid, label=label, kind="python", path=rel)

    def add_io_edge(source: str, target: str, ek: Literal["input", "output"]) -> None:
        key = (source, target, ek)
        if key in io_seen:
            return
        io_seen.add(key)
        nonlocal edge_i
        eid = f"e{edge_i}"
        edge_i += 1
        edges.append(GraphEdge(id=eid, source=source, target=target, kind=ek))

    for nb_rel in notebooks:
        entry = manifest_codes.get(nb_rel, CodeEntry(path=nb_rel))
        parsed_in, parsed_out = parse_notebook_io(root, nb_rel)
        ins, outs = _merge_io(root, entry, parsed_in, parsed_out)
        rt = (runtime_io or {}).get(nb_rel, {})
        rt_ins = {normalize_rel(root, x) for x in rt.get("inputs", [])} if rt else set()
        rt_outs = {normalize_rel(root, x) for x in rt.get("outputs", [])} if rt else set()
        ins = ins | {x for x in rt_ins if x}
        outs = outs | {x for x in rt_outs if x}

        ensure_notebook(nb_rel)
        for d in ins:
            ensure_data(d)
            add_io_edge(data_id(d), code_id(nb_rel), "input")
        for d in outs:
            ensure_data(d)
            add_io_edge(code_id(nb_rel), data_id(d), "output")

        for py_rel in parse_notebook_py_imports_ordered(root, nb_rel, module_map):
            ensure_python(py_rel)
            eid = f"e{edge_i}"
            edge_i += 1
            edges.append(
                GraphEdge(
                    id=eid,
                    source=code_id(nb_rel),
                    target=code_id(py_rel),
                    kind="notebook_call",
                )
            )

    for py_rel in sorted(py_used):
        entry = manifest_codes.get(py_rel, CodeEntry(path=py_rel))
        parsed_in, parsed_out = parse_python_file_io(root, py_rel)
        ins, outs = _merge_io(root, entry, parsed_in, parsed_out)
        rt = (runtime_io or {}).get(py_rel, {})
        rt_ins = {normalize_rel(root, x) for x in rt.get("inputs", [])} if rt else set()
        rt_outs = {normalize_rel(root, x) for x in rt.get("outputs", [])} if rt else set()
        ins = ins | {x for x in rt_ins if x}
        outs = outs | {x for x in rt_outs if x}

        ensure_python(py_rel)
        for d in ins:
            ensure_data(d)
            add_io_edge(data_id(d), code_id(py_rel), "input")
        for d in outs:
            ensure_data(d)
            add_io_edge(code_id(py_rel), data_id(d), "output")

    return GraphPayload(nodes=list(nodes.values()), edges=edges)
