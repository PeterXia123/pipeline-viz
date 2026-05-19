from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from pipeline_viz.models import GraphEdge, GraphNode


def _longest_ranks_io(
    node_ids: list[str],
    io_edges: list[GraphEdge],
) -> dict[str, int]:
    G = nx.DiGraph()
    for n in node_ids:
        G.add_node(n)
    for e in io_edges:
        G.add_edge(e.source, e.target)

    rank: dict[str, int] = {n: 0 for n in G.nodes()}
    try:
        for u in nx.topological_sort(G):
            for v in G.successors(u):
                nv = rank[u] + 1
                if rank[v] < nv:
                    rank[v] = nv
    except nx.NetworkXUnfeasible:
        pass
    return rank


def compute_layout(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> dict[str, Any]:
    """
    - I/O 分层在「不含 notebook 节点」的子图上计算，避免 notebook 与 source data 同列重叠。
    - 若图中存在 notebook：列 0 = 仅 source data；列 1 = notebook；rank>=1 的节点列号 = rank+1。
    - notebook 的 y 置于其 notebook_call 目标 py 的 y 范围中心；notebook_spans 记录与 py 纵向占用一致的高度（布局坐标系）。
    """
    node_map = {n.id: n for n in nodes}
    node_ids = [n.id for n in nodes]
    io_edges = [e for e in edges if e.kind in ("input", "output")]
    call_edges = [e for e in edges if e.kind == "notebook_call"]

    io_only_ids = [nid for nid in node_ids if node_map[nid].kind != "notebook"]
    ranks = _longest_ranks_io(io_only_ids, io_edges)

    nb_ids = [nid for nid in node_ids if node_map[nid].kind == "notebook"]
    nb_set = set(nb_ids)
    has_notebook = len(nb_ids) > 0

    # notebook 直接写出的 data：在「同时存在 notebook_call」时右移一列（与旧逻辑一致）
    nb_sources = {e.source for e in call_edges}
    for nb in nb_sources:
        if nb not in nb_set:
            continue
        has_py = any(e.source == nb for e in call_edges)
        if not has_py:
            continue
        r_nb = ranks.get(nb, 0)
        for e in io_edges:
            if e.source == nb and e.kind == "output":
                tid = e.target
                if tid in ranks:
                    ranks[tid] = max(ranks.get(tid, 0), r_nb + 2)

    for e in call_edges:
        nb, py = e.source, e.target
        rnb = ranks.get(nb, 0)
        ranks[py] = max(ranks.get(py, 0), rnb + 1)

    call_order: dict[str, int] = {}
    for i, e in enumerate(call_edges):
        call_order.setdefault(e.target, i)

    def column_for(nid: str) -> int:
        n = node_map[nid]
        if n.kind == "notebook":
            return 1
        r = ranks.get(nid, 0)
        if not has_notebook:
            return r
        if r == 0:
            return 0
        return r + 1

    def sort_key_non_nb(nid: str) -> tuple:
        n = node_map[nid]
        if n.kind == "python":
            return (0, call_order.get(nid, 9999), n.path)
        return (1, 0, n.path)

    x_step = 220.0
    y_gap = 100.0
    pos: dict[str, tuple[float, float]] = {}
    columns: dict[int, list[str]] = defaultdict(list)

    for nid in node_ids:
        if node_map[nid].kind == "notebook":
            continue
        columns[column_for(nid)].append(nid)

    for col in columns:
        columns[col].sort(key=sort_key_non_nb)

    for col, ids in sorted(columns.items()):
        n = len(ids)
        for i, nid in enumerate(ids):
            x = col * x_step
            y = (i - (n - 1) / 2.0) * y_gap
            pos[nid] = (x, y)

    notebook_spans: dict[str, dict[str, float]] = {}

    for nb_id in nb_ids:
        py_ids = [e.target for e in call_edges if e.source == nb_id]
        py_ys = [pos[pid][1] for pid in py_ids if pid in pos]
        if py_ys:
            cy = (min(py_ys) + max(py_ys)) / 2.0
            pos[nb_id] = (1.0 * x_step, cy)
            span = max(py_ys) - min(py_ys)
            notebook_spans[nb_id] = {
                "span_y": span,
                "padding_y": float(y_gap),
                "box_height": max(span + y_gap, y_gap),
            }
        else:
            pos[nb_id] = (1.0 * x_step, 0.0)
            notebook_spans[nb_id] = {
                "span_y": 0.0,
                "padding_y": float(y_gap),
                "box_height": float(y_gap),
            }

    return {"positions": pos, "notebook_spans": notebook_spans}


def layered_positions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> dict[str, tuple[float, float]]:
    return compute_layout(nodes, edges)["positions"]
