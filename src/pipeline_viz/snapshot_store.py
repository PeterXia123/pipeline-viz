from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from pipeline_viz.models import GraphPayload
from pipeline_viz.notebook_sanitize import raw_file_sha256, stable_ipynb_sha256
from pipeline_viz.paths import code_id, data_id, is_under_root

_SAFE_SNAPSHOT_ID = re.compile(r"^[0-9A-Za-z_.\-]+$")

_SNAPSHOT_CACHE: dict[str, SnapshotRecord] = {}
_SNAPSHOT_CACHE_MAX = 256


def _cache_key(project_root: Path, snapshot_id: str) -> str:
    return f"{project_root.resolve()}\0{snapshot_id}"


def _cache_put(project_root: Path, snapshot_id: str, rec: SnapshotRecord) -> None:
    if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX:
        _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)), None)
    _SNAPSHOT_CACHE[_cache_key(project_root, snapshot_id)] = rec


def _cache_drop(project_root: Path, snapshot_id: str) -> None:
    _SNAPSHOT_CACHE.pop(_cache_key(project_root, snapshot_id), None)


class SnapshotRecord(BaseModel):
    snapshot_id: str = ""
    saved_at: str
    label: str = ""
    description: str = ""
    payload: GraphPayload
    file_hashes: dict[str, str] = Field(default_factory=dict)
    node_labels: dict[str, dict[str, str]] = Field(default_factory=dict)
    data_diffs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if not out.get("snapshot_id"):
            out["snapshot_id"] = str(out.get("saved_at", "legacy")).replace(":", "-")[:48]
        if "file_hashes" not in out:
            out["file_hashes"] = {}
        if "label" not in out:
            out["label"] = ""
        if "description" not in out:
            out["description"] = ""
        if "node_labels" not in out:
            out["node_labels"] = {}
        if "data_diffs" not in out:
            out["data_diffs"] = {}
        return out


def store_dir(project_root: Path) -> Path:
    return project_root / ".pipeline-viz"


def history_root(project_root: Path) -> Path:
    return store_dir(project_root) / "history"


def index_path(project_root: Path) -> Path:
    return history_root(project_root) / "index.json"


def latest_path(project_root: Path) -> Path:
    return store_dir(project_root) / "latest.json"


def snapshot_json_path(project_root: Path, snapshot_id: str) -> Path:
    return history_root(project_root) / f"{snapshot_id}.json"


def snapshot_files_dir(project_root: Path, snapshot_id: str) -> Path:
    return history_root(project_root) / snapshot_id / "files"


def _safe_rel(rel: str) -> str:
    rel = rel.replace("\\", "/").strip("/")
    parts = Path(rel).parts
    for p in parts:
        if p == ".." or p.startswith("/"):
            raise ValueError("invalid path")
    return rel


def blob_path(project_root: Path, snapshot_id: str, rel: str) -> Path:
    rel = _safe_rel(rel)
    return snapshot_files_dir(project_root, snapshot_id) / Path(rel)


def tracked_paths(project_root: Path, payload: GraphPayload) -> list[str]:
    """仅跟踪图中节点对应路径快照，不含 manifest 文件。"""
    return sorted({n.path for n in payload.nodes})


file_sha256 = raw_file_sha256


def compute_file_hashes(project_root: Path, paths: list[str]) -> dict[str, str]:
    root = project_root.resolve()
    out: dict[str, str] = {}
    for rel in paths:
        p = (root / rel).resolve()
        if not is_under_root(root, p):
            continue
        if p.is_file():
            out[rel] = (
                stable_ipynb_sha256(p)
                if rel.lower().endswith(".ipynb")
                else file_sha256(p)
            )
        else:
            out[rel] = ""
    return out


def _payload_fingerprint_str(payload: GraphPayload) -> str:
    """
    图结构的语义指纹（稳定）：
    - 节点按 id/kind/path/label 比较并排序；
    - 边仅按 source/target/kind 比较并排序；
    - 忽略 GraphEdge.id 与列表顺序抖动，避免“无语义变化”误判为新快照。
    """
    nodes = sorted(
        (
            {
                "id": n.id,
                "kind": n.kind,
                "path": n.path,
                "label": n.label,
            }
            for n in payload.nodes
        ),
        key=lambda x: (x["id"], x["kind"], x["path"], x["label"]),
    )
    edges = sorted(
        (
            {
                "source": e.source,
                "target": e.target,
                "kind": e.kind,
            }
            for e in payload.edges
        ),
        key=lambda x: (x["source"], x["target"], x["kind"]),
    )
    return json.dumps(
        {"nodes": nodes, "edges": edges},
        sort_keys=True,
        ensure_ascii=False,
    )


def _hashes_fingerprint_str(hashes: dict[str, str]) -> str:
    return json.dumps(sorted((k, v) for k, v in hashes.items() if v), ensure_ascii=False)


def state_fingerprint(payload: GraphPayload, hashes: dict[str, str]) -> str:
    payload_json = _payload_fingerprint_str(payload)
    hashes_json = _hashes_fingerprint_str(hashes)
    return hashlib.sha256((payload_json + "\n" + hashes_json).encode("utf-8")).hexdigest()


def _read_index(project_root: Path) -> list[dict[str, Any]]:
    p = index_path(project_root)
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_index(project_root: Path, entries: list[dict[str, Any]]) -> None:
    index_path(project_root).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_history(project_root: Path) -> list[dict[str, Any]]:
    """时间顺序：旧 → 新。"""
    return _read_index(project_root)


def list_history_desc(project_root: Path) -> list[dict[str, Any]]:
    return list(reversed(list_history(project_root)))


def load_snapshot(project_root: Path, snapshot_id: str) -> SnapshotRecord | None:
    ck = _cache_key(project_root, snapshot_id)
    cached = _SNAPSHOT_CACHE.get(ck)
    if cached is not None:
        return cached
    p = snapshot_json_path(project_root, snapshot_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rec = SnapshotRecord.model_validate(data)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    _cache_put(project_root, snapshot_id, rec)
    return rec


def load_latest(project_root: Path) -> SnapshotRecord | None:
    lp = latest_path(project_root)
    if not lp.is_file():
        return None
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
        return SnapshotRecord.model_validate(data)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def previous_snapshot_id(project_root: Path, snapshot_id: str) -> Optional[str]:
    entries = list_history(project_root)
    for i, e in enumerate(entries):
        if e.get("snapshot_id") == snapshot_id:
            if i > 0:
                return str(entries[i - 1].get("snapshot_id"))
            return None
    return None


def _copy_tracked_files(project_root: Path, snapshot_id: str, paths: list[str]) -> dict[str, str]:
    root = project_root.resolve()
    hashes: dict[str, str] = {}
    for rel in paths:
        src = (root / rel).resolve()
        if not is_under_root(root, src) or not src.is_file():
            hashes[rel] = ""
            continue
        dst = blob_path(project_root, snapshot_id, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[rel] = (
            stable_ipynb_sha256(src) if rel.lower().endswith(".ipynb") else file_sha256(src)
        )
    return hashes


def save_snapshot(
    project_root: Path, payload: GraphPayload
) -> tuple[str | None, Path | None, str, dict[str, Any] | None]:
    """
    若与上一全局快照状态完全一致，则不写入。
    比较内容：
    - 图结构：节点、边（含 notebook_call 顺序、input/output 关系）；
    - 跟踪文件内容哈希（manifest、各节点路径对应文件等）。

    返回 (snapshot_id, json_path, message, skip_detail)；
    跳过时 snapshot_id 为 None，skip_detail 含 graph_changed / files_changed 供排查。
    """
    sd = store_dir(project_root)
    hr = history_root(project_root)
    sd.mkdir(parents=True, exist_ok=True)
    hr.mkdir(parents=True, exist_ok=True)

    last = load_latest(project_root)

    # 计算本次快照的“跟踪文件集合”
    paths_cur = tracked_paths(project_root, payload)
    hashes_cur = compute_file_hashes(project_root, paths_cur)

    if last:
        prev_hashes_for_compare = _effective_hashes(
            project_root, last.snapshot_id, last
        )
        if not prev_hashes_for_compare:
            paths_prev = tracked_paths(project_root, last.payload)
            paths_union = sorted(set(paths_cur) | set(paths_prev))
            prev_hashes_for_compare = compute_file_hashes(project_root, paths_union)

        graph_changed = _payload_fingerprint_str(last.payload) != _payload_fingerprint_str(payload)
        files_changed = _hashes_fingerprint_str(prev_hashes_for_compare) != _hashes_fingerprint_str(
            hashes_cur
        )

        if not graph_changed and not files_changed:
            return None, None, "no_changes", {
                "graph_changed": False,
                "files_changed": False,
                "nodes": len(payload.nodes),
                "edges": len(payload.edges),
            }

    sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z_" + uuid.uuid4().hex[:8]
    file_hashes = _copy_tracked_files(project_root, sid, paths_cur)

    rec = SnapshotRecord(
        snapshot_id=sid,
        saved_at=datetime.now(timezone.utc).isoformat(),
        label="",
        description="",
        payload=payload,
        file_hashes=file_hashes,
    )

    jp = snapshot_json_path(project_root, sid)
    jp.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    latest_path(project_root).write_text(rec.model_dump_json(indent=2), encoding="utf-8")

    idx = _read_index(project_root)
    idx.append(
        {
            "snapshot_id": sid,
            "saved_at": rec.saved_at,
            "label": rec.label,
            "description": rec.description,
        }
    )
    _write_index(project_root, idx)

    _cleanup_old_blobs(project_root, sid)

    return sid, jp, "saved", None


def _cleanup_old_blobs(project_root: Path, keep_sid: str) -> None:
    hr = history_root(project_root)
    if not hr.is_dir():
        return
    for child in hr.iterdir():
        if not child.is_dir() or child.name == keep_sid:
            continue
        files_dir = child / "files"
        if files_dir.is_dir():
            shutil.rmtree(files_dir, ignore_errors=True)


def delete_snapshot(project_root: Path, snapshot_id: str) -> bool:
    """
    删除某次全局快照：索引项、{id}.json、{id}/files 下文件副本一并删除。
    若删除的是当前 latest，则将 latest 指向剩余快照中最新一条；若无剩余则删除 latest.json。
    """
    if not snapshot_id or len(snapshot_id) > 256 or not _SAFE_SNAPSHOT_ID.match(snapshot_id):
        return False
    idx = _read_index(project_root)
    new_idx = [e for e in idx if e.get("snapshot_id") != snapshot_id]
    if len(new_idx) == len(idx):
        return False

    hr = history_root(project_root)
    jp = hr / f"{snapshot_id}.json"
    snap_dir = hr / snapshot_id
    if jp.is_file():
        jp.unlink()
    if snap_dir.is_dir():
        shutil.rmtree(snap_dir, ignore_errors=True)

    _cache_drop(project_root, snapshot_id)
    _write_index(project_root, new_idx)

    latest = load_latest(project_root)
    if latest and latest.snapshot_id == snapshot_id:
        if new_idx:
            new_latest_id = str(new_idx[-1]["snapshot_id"])
            rec = load_snapshot(project_root, new_latest_id)
            if rec:
                latest_path(project_root).write_text(rec.model_dump_json(indent=2), encoding="utf-8")
            elif latest_path(project_root).is_file():
                latest_path(project_root).unlink()
        else:
            if latest_path(project_root).is_file():
                latest_path(project_root).unlink()
    return True


def set_snapshot_meta(
    project_root: Path, snapshot_id: str, *, label: str, description: str
) -> bool:
    rec = load_snapshot(project_root, snapshot_id)
    if not rec:
        return False
    rec = rec.model_copy(update={"label": label, "description": description})
    snapshot_json_path(project_root, snapshot_id).write_text(
        rec.model_dump_json(indent=2), encoding="utf-8"
    )
    _cache_drop(project_root, snapshot_id)
    latest = load_latest(project_root)
    if latest and latest.snapshot_id == snapshot_id:
        latest_path(project_root).write_text(rec.model_dump_json(indent=2), encoding="utf-8")

    idx = _read_index(project_root)
    for e in idx:
        if e.get("snapshot_id") == snapshot_id:
            e["label"] = label
            e["description"] = description
            break
    _write_index(project_root, idx)
    return True


def set_snapshot_label(project_root: Path, snapshot_id: str, label: str) -> bool:
    rec = load_snapshot(project_root, snapshot_id)
    if not rec:
        return False
    return set_snapshot_meta(
        project_root,
        snapshot_id,
        label=label,
        description=rec.description or "",
    )


def set_node_meta(
    project_root: Path,
    snapshot_id: str,
    node_path: str,
    *,
    label: str,
    description: str,
) -> bool:
    rec = load_snapshot(project_root, snapshot_id)
    if not rec:
        return False
    nl = dict(rec.node_labels)
    nl[node_path] = {"label": label, "description": description}
    rec = rec.model_copy(update={"node_labels": nl})
    snapshot_json_path(project_root, snapshot_id).write_text(
        rec.model_dump_json(indent=2), encoding="utf-8"
    )
    _cache_drop(project_root, snapshot_id)
    latest = load_latest(project_root)
    if latest and latest.snapshot_id == snapshot_id:
        latest_path(project_root).write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    return True


def node_path_from_graph_id(node_id: str) -> Optional[str]:
    if node_id.startswith("data:"):
        return node_id[len("data:") :]
    if node_id.startswith("code:"):
        return node_id[len("code:") :]
    return None


def history_for_node(
    project_root: Path, node_path: str, *, is_data: bool = False
) -> list[dict[str, Any]]:
    """某路径的「文件版本点」。

    仅保留真正的版本节点：首个可追踪版本 + 内容哈希变化的版本。
    - data：要求快照时磁盘上确有该文件（非空 hash）才开始计入。
    - code：优先按 file_hashes 追踪；若旧快照无 hash 且图中出现该路径，保留一次兜底版本。
    """
    out_asc: list[dict[str, Any]] = []
    last_hash: str | None = None
    seen_version = False
    emitted_code_fallback = False
    for meta in list_history(project_root):
        sid = meta.get("snapshot_id")
        if not sid:
            continue
        rec = load_snapshot(project_root, sid)
        if not rec:
            continue
        in_graph = any(n.path == node_path for n in rec.payload.nodes)
        h = (rec.file_hashes.get(node_path) or "").strip()
        if is_data:
            if not h:
                continue
            if h == last_hash:
                continue
            last_hash = h
            seen_version = True
        else:
            if not (in_graph or h):
                continue
            if h:
                if h == last_hash:
                    continue
                last_hash = h
                seen_version = True
            elif seen_version:
                # code 进入可追踪阶段后，跳过无 hash 快照，避免把每次全局保存都算成该文件新版本
                continue
            elif emitted_code_fallback:
                # 仅在无法拿到 hash 的历史阶段保留一次兜底记录
                continue
            elif not in_graph:
                continue
            else:
                emitted_code_fallback = True
        nl = rec.node_labels.get(node_path, {})
        out_asc.append(
            {
                "snapshot_id": sid,
                "saved_at": rec.saved_at,
                "label": nl.get("label") or "",
                "description": nl.get("description") or "",
            }
        )
    return sorted(out_asc, key=lambda x: x["saved_at"], reverse=True)


def read_blob(project_root: Path, snapshot_id: str, rel: str) -> bytes | None:
    p = blob_path(project_root, snapshot_id, rel)
    if not p.is_file():
        return None
    try:
        return p.read_bytes()
    except OSError:
        return None


def _node_id_for_path(path: str) -> str:
    # manifest yaml uses data: or we use code: for code files
    low = path.lower()
    if low.endswith((".yaml", ".yml")) and path in ("pipeline-manifest.yaml", "manifest.yaml"):
        return data_id(path)
    if path.endswith(".ipynb") or path.endswith(".py"):
        return code_id(path)
    return data_id(path)


def _graph_node_id_for_tracked_rel(rel: str, current_payload: GraphPayload) -> Optional[str]:
    """
    将 file_hashes 中的相对路径映射到当前图中的节点 id。
    manifest / 磁盘上的写法可能与构图时略有差异（./ 前缀、分隔符等），故做多级回退。
    """
    nid = _node_id_for_path(rel)
    cur_ids = {n.id for n in current_payload.nodes}
    if nid in cur_ids:
        return nid
    rel_norm = rel.replace("\\", "/").lstrip("/")
    if rel_norm.startswith("./"):
        rel_norm = rel_norm[2:]
    for n in current_payload.nodes:
        np = n.path.replace("\\", "/").lstrip("/")
        if np.startswith("./"):
            np = np[2:]
        if np == rel_norm or np.endswith("/" + rel_norm):
            return n.id
    base = Path(rel).name
    if base:
        for n in current_payload.nodes:
            if Path(n.path.replace("\\", "/")).name == base:
                return n.id
    return None


def _compute_hashes_from_blobs(
    project_root: Path, snapshot_id: str, payload: GraphPayload
) -> dict[str, str]:
    """Legacy snapshots lack file_hashes; recompute from stored blob copies."""
    paths = tracked_paths(project_root, payload)
    out: dict[str, str] = {}
    for rel in paths:
        bp = blob_path(project_root, snapshot_id, rel)
        if bp.is_file():
            out[rel] = (
                stable_ipynb_sha256(bp)
                if rel.lower().endswith(".ipynb")
                else file_sha256(bp)
            )
    return out


def _effective_hashes(
    project_root: Path, snapshot_id: str, rec: SnapshotRecord
) -> dict[str, str]:
    recomputed = _compute_hashes_from_blobs(project_root, snapshot_id, rec.payload)
    if recomputed:
        return recomputed
    if rec.file_hashes:
        return rec.file_hashes
    return {}


def diff_highlights(
    current: GraphPayload,
    previous: GraphPayload | None,
) -> dict[str, str]:
    """当前图相对 previous 快照：新增节点 / 新边端点。"""
    if previous is None:
        return {}

    prev_nodes = {n.id for n in previous.nodes}
    prev_edges = {(e.source, e.target, e.kind) for e in previous.edges}
    cur_nodes = {n.id for n in current.nodes}
    cur_edges = {(e.source, e.target, e.kind) for e in current.edges}

    hi: dict[str, str] = {}
    for nid in cur_nodes - prev_nodes:
        hi[nid] = "#b7f5c8"
    for s, t, k in cur_edges - prev_edges:
        hi.setdefault(s, "#d4edda")
        hi.setdefault(t, "#d4edda")
    return hi


def diff_edge_highlights(
    current: GraphPayload,
    previous: GraphPayload | None,
) -> dict[str, str]:
    """当前图相对 previous：新增的边高亮（key: 'source|target|kind'）。"""
    if previous is None:
        return {}
    prev_edges = {(e.source, e.target, e.kind) for e in previous.edges}
    cur_edges = {(e.source, e.target, e.kind) for e in current.edges}
    out: dict[str, str] = {}
    for s, t, k in cur_edges - prev_edges:
        out[f"{s}|{t}|{k}"] = "#2e7d32"  # 绿色：相对最新快照新增连线
    return out


def commit_highlights(
    project_root: Path,
    snapshot_id: str,
    current_payload: GraphPayload,
) -> dict[str, str]:
    """
    选中某次全局快照 S 时：相对上一快照 S-1，在本次提交中发生变化的文件 → 高亮当前图中对应节点。
    """
    rec_sel = load_snapshot(project_root, snapshot_id)
    if not rec_sel:
        return {}

    prev_id = previous_snapshot_id(project_root, snapshot_id)
    rec_prev = load_snapshot(project_root, prev_id) if prev_id else None

    # 第一条全局快照：没有上一版可比，无「相对上一版的变更」语义，不标红（与后续快照区分）
    if rec_prev is None:
        return {}

    prev_hashes = _effective_hashes(project_root, prev_id, rec_prev)
    sel_hashes = _effective_hashes(project_root, snapshot_id, rec_sel)

    all_paths = set(prev_hashes) | set(sel_hashes)
    changed_paths: set[str] = set()
    for p in all_paths:
        if prev_hashes.get(p) != sel_hashes.get(p):
            changed_paths.add(p)

    cur_ids = {n.id for n in current_payload.nodes}
    hi: dict[str, str] = {}
    for rel in changed_paths:
        prev_h = prev_hashes.get(rel) or ""
        sel_h = sel_hashes.get(rel) or ""
        if not prev_h and not sel_h:
            continue
        if not prev_h and sel_h:
            # 文件首次生成：虚线→实线即可，不标红
            continue
        nid = _graph_node_id_for_tracked_rel(rel, current_payload)
        if nid:
            hi[nid] = "#ffb3b3"

    return hi


def commit_edge_highlights(
    project_root: Path,
    snapshot_id: str,
    current_payload: GraphPayload,
) -> dict[str, str]:
    """选中某次全局快照时：相对上一快照，新增的边高亮（key: 'source|target|kind'）。"""
    rec_sel = load_snapshot(project_root, snapshot_id)
    if not rec_sel:
        return {}
    prev_id = previous_snapshot_id(project_root, snapshot_id)
    rec_prev = load_snapshot(project_root, prev_id) if prev_id else None
    if rec_prev is None:
        # 第一条：无上一条快照可比，不标红边
        return {}
    out: dict[str, str] = {}
    prev_edges = {(e.source, e.target, e.kind) for e in rec_prev.payload.edges}
    cur_edges = {(e.source, e.target, e.kind) for e in rec_sel.payload.edges}
    for s, t, k in cur_edges - prev_edges:
        out[f"{s}|{t}|{k}"] = "#b71c1c"

    return out


def path_changed_in_commit(project_root: Path, snapshot_id: str, rel: str) -> bool:
    """
    与 commit_highlights 一致：rel 在该次全局快照相对上一快照是否发生文件内容变化。
    用于「选中全局版本时」筛选右侧单文件历史。
    """
    rec_sel = load_snapshot(project_root, snapshot_id)
    if not rec_sel:
        return False
    prev_id = previous_snapshot_id(project_root, snapshot_id)
    rec_prev = load_snapshot(project_root, prev_id) if prev_id else None

    sel_hashes = _effective_hashes(project_root, snapshot_id, rec_sel)
    if rec_prev is None:
        return bool(sel_hashes.get(rel))

    prev_hashes = _effective_hashes(project_root, prev_id, rec_prev)
    if prev_hashes.get(rel) != sel_hashes.get(rel):
        if not (prev_hashes.get(rel) or sel_hashes.get(rel)):
            return False
        return True
    return False


def save_latest(project_root: Path, payload: GraphPayload) -> Path:
    """兼容旧调用：总是返回 latest.json 路径（即使本次未保存）。"""
    save_snapshot(project_root, payload)
    return latest_path(project_root)
