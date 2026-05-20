from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline_viz.data_profile import (
    compute_schema_diff,
    extract_profile,
    extract_profile_from_bytes,
    is_dataframe_format,
)
from pipeline_viz.dataframe_compare import compare_snapshot_blob_to_disk, is_dataframe_file
from pipeline_viz.file_diff import diff_pair_for_display, read_text_safe
from pipeline_viz.graph_builder import build_graph
from pipeline_viz.layout import compute_layout
from pipeline_viz.models import GraphPayload
from pipeline_viz.notebook_sanitize import sanitize_notebook_dict
from pipeline_viz.io_tracer import (
    filter_project_paths,
    generate_trace_collect_code,
    generate_trace_startup_code,
    load_runtime_io,
    parse_trace_output,
    update_runtime_io,
)
from pipeline_viz.paths import SCAN_SKIP_DIR_NAMES, is_under_root
from pipeline_viz.snapshot_store import (
    commit_highlights,
    commit_edge_highlights,
    delete_snapshot,
    diff_highlights,
    diff_edge_highlights,
    history_for_node,
    list_history,
    list_history_desc,
    load_latest,
    load_snapshot,
    node_path_from_graph_id,
    path_changed_in_commit,
    previous_snapshot_id,
    read_blob,
    save_snapshot,
    set_node_meta,
    set_snapshot_label,
    set_snapshot_meta,
)

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="pipeline-viz")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC)), name="assets")

_API_TOKEN = os.environ.get("PIPELINE_VIZ_API_TOKEN", "").strip()


@app.middleware("http")
async def _check_api_token(request, call_next):
    if _API_TOKEN and request.url.path.startswith("/api/"):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {_API_TOKEN}":
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


class _GraphCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[dict[str, float], GraphPayload]] = {}

    def get_or_build(self, root: Path, manifest_path: Optional[Path]) -> GraphPayload:
        key = str(root.resolve())
        fps = self._scan_mtimes(root, manifest_path)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == fps:
            return cached[1]
        runtime_io = load_runtime_io(root)
        payload = build_graph(root, manifest_path, runtime_io=runtime_io)
        self._entries[key] = (fps, payload)
        return payload

    def invalidate(self, root: Path) -> None:
        self._entries.pop(str(root.resolve()), None)

    @staticmethod
    def _scan_mtimes(root: Path, manifest_path: Optional[Path]) -> dict[str, float]:
        fps: dict[str, float] = {}
        if manifest_path and manifest_path.is_file():
            fps[str(manifest_path)] = manifest_path.stat().st_mtime
        rio = root / ".pipeline-viz" / "runtime_io.json"
        if rio.is_file():
            try:
                fps[str(rio)] = rio.stat().st_mtime
            except OSError:
                pass
        for pattern in ("*.ipynb", "*.py"):
            for p in root.rglob(pattern):
                if any(part in SCAN_SKIP_DIR_NAMES for part in p.parts):
                    continue
                try:
                    fps[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        return fps


_graph_cache = _GraphCache()


def _resolve_root(project_root: Optional[str]) -> Path:
    if project_root:
        return Path(project_root).expanduser().resolve()
    env = os.environ.get("PIPELINE_VIZ_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd()


def _blob_utf8_text(project_root: Path, snapshot_id: str, rel: str) -> str:
    b = read_blob(project_root, snapshot_id, rel)
    if b is None:
        return ""
    return b.decode("utf-8", errors="replace")


def _manifest_path(root: Path) -> Optional[Path]:
    for name in ("pipeline-manifest.yaml", "manifest.yaml"):
        p = root / name
        if p.is_file():
            return p
    return None


def _env_for_notebook_run(project_root: Path) -> dict[str, str]:
    """让 notebook 内能 import 项目根下的包（如 pipeline、lib），并保留原有 PATH。"""
    env = os.environ.copy()
    root_s = str(project_root.resolve())
    extra = (env.get("PIPELINE_VIZ_PYTHONPATH_EXTRA") or "").strip()
    parts: list[str] = [p for p in (root_s, extra) if p]
    old = env.get("PYTHONPATH", "").strip()
    if old:
        parts.append(old)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _build_graph_or_400(root: Path):
    try:
        return _graph_cache.get_or_build(root, _manifest_path(root))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"无法解析项目图，请检查 manifest/YAML 或代码路径配置: {e}",
        ) from e


def _run_notebook_execute_inplace(project_root: Path, nb_path: Path) -> tuple[set[str], set[str]]:
    try:
        import nbformat
        from nbclient.exceptions import CellExecutionError
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="执行 notebook 需要 nbformat、nbconvert、ipykernel。",
        ) from e

    timeout = int(os.environ.get("PIPELINE_VIZ_RUN_TIMEOUT", "600"))
    startup_timeout = int(os.environ.get("PIPELINE_VIZ_KERNEL_STARTUP_TIMEOUT", "120"))
    env = _env_for_notebook_run(project_root)

    with nb_path.open(encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    startup_cell = nbformat.v4.new_code_cell(
        generate_trace_startup_code(str(project_root.resolve()))
    )
    startup_cell.metadata["pipeline_viz_trace"] = True
    collect_cell = nbformat.v4.new_code_cell(generate_trace_collect_code())
    collect_cell.metadata["pipeline_viz_trace"] = True

    notebook.cells.insert(0, startup_cell)
    notebook.cells.append(collect_cell)

    ep = ExecutePreprocessor(
        timeout=timeout,
        startup_timeout=startup_timeout,
        kernel_manager_class="jupyter_client.manager.KernelManager",
    )
    resources = {"metadata": {"path": str(project_root.resolve())}}
    ep.km_kwargs = {"env": env}
    try:
        ep.preprocess(notebook, resources)
    except CellExecutionError as e:
        detail = str(e).strip()
        if len(detail) > 4000:
            detail = detail[:4000] + "\n…(truncated)"
        raise HTTPException(status_code=500, detail=detail) from e

    traced_reads, traced_writes = parse_trace_output(
        notebook.cells[-1].get("outputs", [])
    )
    traced_inputs = filter_project_paths(project_root, traced_reads)
    traced_outputs = filter_project_paths(project_root, traced_writes)

    notebook.cells = [
        c for c in notebook.cells if not c.metadata.get("pipeline_viz_trace")
    ]

    data = json.loads(nbformat.writes(notebook))
    sanitize_notebook_dict(data)
    notebook = nbformat.from_dict(data)
    with nb_path.open("w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return traced_inputs, traced_outputs


@app.get("/")
def index():
    index_html = STATIC / "index.html"
    if not index_html.is_file():
        return JSONResponse({"detail": "static UI missing"}, status_code=500)
    return FileResponse(
        index_html,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/graph")
def api_graph(
    project_root: Optional[str] = Query(default=None),
    view_snapshot_id: Optional[str] = Query(
        default=None,
        description="选中某次全局快照时，高亮该次提交相对上一版本变更的文件",
    ),
):
    root = _resolve_root(project_root)
    if not root.is_dir():
        raise HTTPException(400, "project_root is not a directory")
    # 默认：展示磁盘当前图；若指定 view_snapshot_id：展示该快照保存下来的图（避免磁盘已变化导致高亮与图不一致）
    if view_snapshot_id:
        rec = load_snapshot(root, view_snapshot_id)
        if not rec:
            raise HTTPException(404, "snapshot not found")
        payload = rec.payload
    else:
        payload = _build_graph_or_400(root)
    missing_node_ids: list[str] = []
    for n in payload.nodes:
        if n.kind not in ("data", "notebook", "python"):
            continue
        p = (root / n.path).resolve()
        if not is_under_root(root, p) or not p.is_file():
            missing_node_ids.append(n.id)
    layout = compute_layout(payload.nodes, payload.edges)
    pos = layout["positions"]
    prev = load_latest(root)

    new_node_ids: list[str] = []
    if view_snapshot_id:
        highlights = commit_highlights(root, view_snapshot_id, payload)
        edge_highlights = commit_edge_highlights(root, view_snapshot_id, payload)
        prev_sid = previous_snapshot_id(root, view_snapshot_id)
        if prev_sid:
            prev_rec = load_snapshot(root, prev_sid)
            prev_node_ids = {n.id for n in prev_rec.payload.nodes} if prev_rec else set()
            new_node_ids = [n.id for n in payload.nodes if n.id not in prev_node_ids and n.kind == "data"]
        else:
            new_node_ids = [n.id for n in payload.nodes if n.kind == "data"]
    else:
        highlights = diff_highlights(payload, prev.payload if prev else None)
        edge_highlights = diff_edge_highlights(payload, prev.payload if prev else None)

    hist = list_history(root)
    dbg_prev_for_view = (
        previous_snapshot_id(root, view_snapshot_id) if view_snapshot_id else None
    )
    mp = _manifest_path(root)
    return {
        "project_root": str(root),
        "manifest": str(mp) if mp else None,
        "graph": payload.model_dump(),
        "positions": {k: [v[0], v[1]] for k, v in pos.items()},
        "notebook_spans": layout["notebook_spans"],
        "highlights": highlights,
        "edge_highlights": edge_highlights,
        "highlight_counts": {
            "node": len(highlights or {}),
            "edge": len(edge_highlights or {}),
        },
        "server_features": {
            "edge_highlights": True,
            "commit_graph_highlights": True,
            "graph_cache": True,
        },
        "parse_note": (
            "静态分析可能遗漏动态路径（如 glob/循环/条件分支中的 I/O）；"
            "可在 pipeline-manifest.yaml 中显式声明补全。"
        ),
        "highlight_debug": {
            "view_snapshot_id": view_snapshot_id,
            "previous_snapshot_id_for_view": dbg_prev_for_view,
            "latest_snapshot_id": prev.snapshot_id if prev else None,
            "using_saved_payload": bool(view_snapshot_id),
        },
        "snapshot_time": prev.saved_at if prev else None,
        "snapshot_id": prev.snapshot_id if prev else None,
        "history_count": len(hist),
        "view_snapshot_id": view_snapshot_id,
        "missing_node_ids": missing_node_ids,
        "new_node_ids": new_node_ids,
    }


@app.post("/api/snapshot/save")
def api_snapshot_save(
    project_root: Optional[str] = Query(default=None),
    body: dict = Body(default_factory=dict),
):
    root = _resolve_root(project_root)
    payload = _build_graph_or_400(root)
    data_diffs = body.get("data_diffs") or {}
    sid, jp, msg, skip_detail = save_snapshot(root, payload, data_diffs=data_diffs)
    if sid is None:
        out = {
            "skipped": True,
            "reason": msg,
            "nodes": len(payload.nodes),
            "edges": len(payload.edges),
            "message": (
                "未保存：相对上一全局快照，图（节点、连线、顺序）与跟踪文件内容均未变化。"
                "保存时已从磁盘重新构图。请检查：① 文件已写入磁盘；② 顶部「项目根目录」指向正在编辑的项目；"
                "③ 修改已体现在 manifest / notebook 源码中（否则构图不变）。"
            ),
        }
        if skip_detail:
            out.update(skip_detail)
        return out
    return {
        "skipped": False,
        "snapshot_id": sid,
        "saved": str(jp) if jp else "",
        "nodes": len(payload.nodes),
        "edges": len(payload.edges),
    }


@app.get("/api/history")
def api_history(project_root: Optional[str] = Query(default=None)):
    root = _resolve_root(project_root)
    return {"items": list_history_desc(root)}


def _delete_snapshot_response(root: Path, snapshot_id: str) -> dict:
    if not delete_snapshot(root, snapshot_id):
        raise HTTPException(404, "snapshot not found or invalid id")
    return {"ok": True, "deleted": snapshot_id}


@app.post("/api/history/delete")
def api_history_delete_post(
    snapshot_id: str = Query(..., description="要删除的全局快照 ID"),
    project_root: Optional[str] = Query(default=None),
):
    """删除全局快照（推荐 POST，与 label/meta 一致；避免部分环境对 DELETE 不友好）。"""
    root = _resolve_root(project_root)
    return _delete_snapshot_response(root, snapshot_id)


@app.delete("/api/history/snapshot")
def api_history_snapshot_delete(
    snapshot_id: str = Query(..., description="要删除的全局快照 ID"),
    project_root: Optional[str] = Query(default=None),
):
    """删除全局快照及其在 .pipeline-viz/history 下保存的文件副本。"""
    root = _resolve_root(project_root)
    return _delete_snapshot_response(root, snapshot_id)


def _apply_snapshot_label(root: Path, snapshot_id: str, label: str) -> dict:
    if not set_snapshot_label(root, snapshot_id, label):
        raise HTTPException(404, "snapshot not found")
    return {"ok": True, "snapshot_id": snapshot_id, "label": label}


def _apply_snapshot_meta(root: Path, snapshot_id: str, label: str, description: str) -> dict:
    if not set_snapshot_meta(root, snapshot_id, label=label, description=description):
        raise HTTPException(404, "snapshot not found")
    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "label": label,
        "description": description,
    }


@app.post("/api/history/label")
def api_history_label_post(
    snapshot_id: str = Query(..., description="全局快照 ID"),
    project_root: Optional[str] = Query(default=None),
    label: str = Body("", embed=True),
):
    """推荐：POST + JSON body `{"label":"..."}`，避免 PATCH 在部分环境下的问题。"""
    root = _resolve_root(project_root)
    return _apply_snapshot_label(root, snapshot_id, label)


@app.patch("/api/history/label")
def api_history_label_patch(
    snapshot_id: str = Query(..., description="全局快照 ID"),
    label: str = Query("", description="显示名称"),
    project_root: Optional[str] = Query(default=None),
):
    root = _resolve_root(project_root)
    return _apply_snapshot_label(root, snapshot_id, label)


@app.post("/api/history/meta")
def api_history_meta_post(
    snapshot_id: str = Query(..., description="全局快照 ID"),
    project_root: Optional[str] = Query(default=None),
    body: dict = Body(default_factory=dict),
):
    """更新快照显示名与描述。"""
    root = _resolve_root(project_root)
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")
    label = rec.label if "label" not in body else str(body.get("label") or "")
    description = (
        rec.description if "description" not in body else str(body.get("description") or "")
    )
    return _apply_snapshot_meta(root, snapshot_id, label, description)


@app.get("/api/node/history")
def api_node_history(
    node_id: str = Query(..., description="如 code:path 或 data:path"),
    project_root: Optional[str] = Query(default=None),
    view_snapshot_id: Optional[str] = Query(
        default=None,
        description="选中某全局快照时：仅展示该次提交中该文件对应的版本；未变更则空列表",
    ),
):
    root = _resolve_root(project_root)
    p = node_path_from_graph_id(node_id)
    if not p:
        raise HTTPException(400, "invalid node_id")
    is_data = node_id.startswith("data:")

    if view_snapshot_id:
        rec = load_snapshot(root, view_snapshot_id)
        if not rec:
            raise HTTPException(404, "snapshot not found")
        if not path_changed_in_commit(root, view_snapshot_id, p):
            return {
                "path": p,
                "entries": [],
                "view_snapshot_id": view_snapshot_id,
                "commit_filtered": True,
            }
        nl = rec.node_labels.get(p, {})
        return {
            "path": p,
            "entries": [
                {
                    "snapshot_id": rec.snapshot_id,
                    "saved_at": rec.saved_at,
                    "label": nl.get("label") or "",
                    "description": nl.get("description") or "",
                }
            ],
            "view_snapshot_id": view_snapshot_id,
            "commit_filtered": True,
        }

    return {
        "path": p,
        "entries": history_for_node(root, p, is_data=is_data),
        "view_snapshot_id": None,
        "commit_filtered": False,
    }


@app.post("/api/node/meta")
def api_node_meta(
    snapshot_id: str = Query(..., description="快照 ID"),
    node_path: str = Query(..., description="文件相对路径"),
    project_root: Optional[str] = Query(default=None),
    body: dict = Body(default_factory=dict),
):
    root = _resolve_root(project_root)
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")
    cur = rec.node_labels.get(node_path, {})
    label = cur.get("label", "") if "label" not in body else str(body.get("label") or "")
    description = cur.get("description", "") if "description" not in body else str(body.get("description") or "")
    if not set_node_meta(root, snapshot_id, node_path, label=label, description=description):
        raise HTTPException(500, "failed to save")
    return {"ok": True, "snapshot_id": snapshot_id, "node_path": node_path, "label": label, "description": description}


MAX_COLUMNS_PREVIEW = 30


@app.get("/api/data/columns")
def api_data_columns(
    path: str = Query(..., description="data 文件相对路径"),
    project_root: Optional[str] = Query(default=None),
):
    root = _resolve_root(project_root)
    fp = (root / path).resolve()
    if not is_under_root(root, fp):
        raise HTTPException(400, "path escapes project root")
    if not fp.is_file():
        return {"path": path, "columns": [], "total": 0, "error": "文件不存在"}

    ext = fp.suffix.lower()
    try:
        if ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            cols = pd.read_csv(fp, nrows=0, sep=sep).columns.tolist()
        elif ext in (".parquet", ".pq"):
            try:
                import pyarrow.parquet as pq
                cols = pq.read_schema(fp).names
            except ImportError:
                import pandas as pd
                cols = pd.read_parquet(fp).columns.tolist()
        elif ext in (".feather", ".ftr", ".arrow", ".ipc"):
            import pyarrow.feather as pf
            cols = pf.read_table(fp, columns=[]).schema.names
        elif ext in (".json", ".jsonl", ".ndjson"):
            import pandas as pd
            if ext == ".json":
                cols = pd.read_json(fp, nrows=1).columns.tolist()
            else:
                cols = pd.read_json(fp, lines=True, nrows=1).columns.tolist()
        else:
            return {"path": path, "columns": [], "total": 0, "error": f"不支持的格式: {ext}"}
    except Exception as e:
        return {"path": path, "columns": [], "total": 0, "error": str(e)[:200]}

    total = len(cols)
    truncated = total > MAX_COLUMNS_PREVIEW
    return {
        "path": path,
        "columns": cols[:MAX_COLUMNS_PREVIEW],
        "total": total,
        "truncated": truncated,
    }


@app.get("/api/file/diff")
def api_file_diff(
    path: str = Query(..., description="相对项目根的路径"),
    snapshot_id: str = Query(..., description="作为「新」侧的快照 id"),
    project_root: Optional[str] = Query(default=None),
    compare: str = Query(
        "previous",
        description="previous: 与上一全局快照对比；current: 该快照 vs 当前磁盘",
    ),
):
    root = _resolve_root(project_root)
    if compare not in ("previous", "current"):
        raise HTTPException(400, "compare must be previous or current")
    rel = path.replace("\\", "/").lstrip("/")
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")

    if compare == "current":
        snap_txt = _blob_utf8_text(root, snapshot_id, rel)
        disk_txt = read_text_safe(root / rel)
        return diff_pair_for_display(rel, snap_txt, disk_txt)

    prev_id = previous_snapshot_id(root, snapshot_id)
    prev_txt = _blob_utf8_text(root, prev_id, rel) if prev_id else ""
    curr_txt = _blob_utf8_text(root, snapshot_id, rel)
    return diff_pair_for_display(rel, prev_txt, curr_txt)


@app.get("/api/file/schema-diff")
def api_file_schema_diff(
    path: str = Query(..., description="相对项目根的路径"),
    snapshot_id: str = Query(..., description="作为「新」侧的快照 id"),
    project_root: Optional[str] = Query(default=None),
    compare: str = Query(
        "previous",
        description="previous: 与上一全局快照对比；current: 该快照 vs 当前磁盘",
    ),
):
    root = _resolve_root(project_root)
    if compare not in ("previous", "current"):
        raise HTTPException(400, "compare must be previous or current")
    rel = path.replace("\\", "/").lstrip("/")
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")

    suffix = Path(rel).suffix

    if compare == "current":
        snap_blob = read_blob(root, snapshot_id, rel)
        old_profile = (
            extract_profile_from_bytes(snap_blob, suffix)
            if snap_blob
            else {"row_count": None, "col_count": None, "columns": [], "dtypes": {}, "file_size": 0, "format": "missing", "error": "no blob"}
        )
        new_profile = extract_profile(root / rel)
    else:
        prev_id = previous_snapshot_id(root, snapshot_id)
        if prev_id:
            prev_blob = read_blob(root, prev_id, rel)
            old_profile = (
                extract_profile_from_bytes(prev_blob, suffix)
                if prev_blob
                else {"row_count": None, "col_count": None, "columns": [], "dtypes": {}, "file_size": 0, "format": "missing", "error": "no blob"}
            )
        else:
            old_profile = {"row_count": None, "col_count": None, "columns": [], "dtypes": {}, "file_size": 0, "format": "new_file", "error": None}
        curr_blob = read_blob(root, snapshot_id, rel)
        new_profile = (
            extract_profile_from_bytes(curr_blob, suffix)
            if curr_blob
            else {"row_count": None, "col_count": None, "columns": [], "dtypes": {}, "file_size": 0, "format": "missing", "error": "no blob"}
        )

    schema_diff = compute_schema_diff(old_profile, new_profile)
    return {
        "old_profile": old_profile,
        "new_profile": new_profile,
        "schema_diff": schema_diff,
        "is_dataframe": is_dataframe_format(rel) and old_profile.get("error") is None and new_profile.get("error") is None,
    }


@app.get("/api/snapshot/latest")
def api_snapshot_latest(project_root: Optional[str] = Query(default=None)):
    root = _resolve_root(project_root)
    prev = load_latest(root)
    if not prev:
        return {"snapshot": None}
    return {"snapshot": prev.model_dump()}


@app.get("/api/snapshot/data-diffs")
def api_snapshot_data_diffs(
    snapshot_id: str = Query(..., description="快照 ID"),
    project_root: Optional[str] = Query(default=None),
):
    root = _resolve_root(project_root)
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")
    return {
        "snapshot_id": snapshot_id,
        "data_diffs": rec.data_diffs,
    }


@app.get("/api/data/datacompy")
def api_data_datacompy(
    path: str = Query(..., description="相对项目根的路径"),
    project_root: Optional[str] = Query(default=None),
    snapshot_id: Optional[str] = Query(
        default=None,
        description="快照 ID；默认使用最新全局快照中的文件作为旧版",
    ),
    merge_keys: str = Query(
        "",
        description="逗号分隔的 merge key；留空则按行顺序比较",
    ),
):
    root = _resolve_root(project_root)
    rel = path.replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(400, "invalid path")
    if not is_dataframe_file(rel):
        raise HTTPException(400, "仅支持 csv / tsv / parquet / pkl 等 DataFrame 文件")

    disk = (root / rel).resolve()
    if not is_under_root(root, disk) or not disk.is_file():
        raise HTTPException(400, "磁盘上不存在该文件")

    latest = load_latest(root)
    sid = snapshot_id or (latest.snapshot_id if latest else None)
    if not sid:
        raise HTTPException(400, "尚无全局快照，请先保存快照")

    blob = read_blob(root, sid, rel)
    if blob is None:
        raise HTTPException(
            404,
            "该快照中未保存此文件（可能因快照时文件不存在或未被跟踪）",
        )

    keys = [k.strip() for k in merge_keys.split(",") if k.strip()]
    result = compare_snapshot_blob_to_disk(blob, rel, disk, keys if keys else None)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=str(result["error"]))
    return result


@app.get("/api/snapshot/review-compare")
def api_snapshot_review_compare(
    path: str = Query(..., description="相对项目根的数据文件路径"),
    project_root: Optional[str] = Query(default=None),
    merge_keys: str = Query("", description="逗号分隔的 merge key"),
):
    root = _resolve_root(project_root)
    rel = path.replace("\\", "/").lstrip("/")
    if not is_dataframe_file(rel):
        raise HTTPException(400, "仅支持 DataFrame 格式文件")

    disk = (root / rel).resolve()
    if not is_under_root(root, disk) or not disk.is_file():
        raise HTTPException(400, "磁盘上不存在该文件")

    latest = load_latest(root)
    if not latest:
        raise HTTPException(400, "尚无全局快照")

    blob = read_blob(root, latest.snapshot_id, rel)
    if blob is None:
        raise HTTPException(404, "该快照中未保存此文件")

    keys = [k.strip() for k in merge_keys.split(",") if k.strip()]
    result = compare_snapshot_blob_to_disk(blob, rel, disk, keys if keys else None)
    if result.get("error"):
        raise HTTPException(400, str(result["error"]))
    return result


@app.get("/api/snapshot/pre-check")
def api_snapshot_precheck(project_root: Optional[str] = Query(default=None)):
    root = _resolve_root(project_root)
    payload = _build_graph_or_400(root)

    from pipeline_viz.snapshot_store import (
        compute_file_hashes,
        tracked_paths,
        _effective_hashes,
        _hashes_fingerprint_str,
        _payload_fingerprint_str,
    )

    last = load_latest(root)
    paths_cur = tracked_paths(root, payload)
    hashes_cur = compute_file_hashes(root, paths_cur)

    if not last:
        return {"has_changes": False, "changed_files": [], "reason": "no_previous_snapshot"}

    prev_hashes = _effective_hashes(root, last.snapshot_id, last)
    graph_changed = _payload_fingerprint_str(last.payload) != _payload_fingerprint_str(payload)
    files_changed = _hashes_fingerprint_str(prev_hashes) != _hashes_fingerprint_str(hashes_cur)

    if not graph_changed and not files_changed:
        return {"has_changes": False, "changed_files": [], "reason": "no_changes"}

    changed_files = []
    # Use data nodes from both the current and previous payload so that files
    # tracked in the last snapshot are checked even if the graph builder didn't
    # re-discover them (e.g. the notebook has no auto-detectable reference).
    data_node_paths = {n.path for n in payload.nodes if n.kind == "data"} | {
        n.path for n in last.payload.nodes if n.kind == "data"
    }

    for rel in sorted(data_node_paths):
        old_h = (prev_hashes.get(rel) or "").strip()
        new_h = (hashes_cur.get(rel) or "").strip()
        if not new_h:
            # Compute current hash directly from disk if not in hashes_cur
            disk_file = (root / rel).resolve()
            if disk_file.is_file():
                from pipeline_viz.snapshot_store import compute_file_hashes as _cfh
                _tmp = _cfh(root, [rel])
                new_h = (_tmp.get(rel) or "").strip()
        if old_h == new_h:
            continue
        if not old_h and new_h:
            continue  # first-time creation, skip

        disk_path = (root / rel).resolve()

        new_profile = extract_profile(disk_path) if disk_path.is_file() else {}
        old_blob = read_blob(root, last.snapshot_id, rel)

        # Determine if actually a readable DataFrame (extension may match but content may not be)
        is_df = is_dataframe_format(rel) and new_profile.get("format") not in (
            "non_dataframe",
            "missing",
        ) and new_profile.get("row_count") is not None

        entry: dict = {
            "path": rel,
            "is_dataframe": is_df,
            "file_size_old": 0,
            "file_size_new": new_profile.get("file_size", 0),
        }

        if old_blob is not None:
            old_profile = extract_profile_from_bytes(old_blob, Path(rel).suffix)
            entry["file_size_old"] = old_profile.get("file_size", 0)
            if is_df:
                entry["schema_diff"] = compute_schema_diff(old_profile, new_profile)
                entry["old_profile"] = old_profile
                entry["new_profile"] = new_profile
        else:
            if is_df:
                entry["schema_diff"] = None
                entry["old_profile"] = None
                entry["new_profile"] = new_profile

        changed_files.append(entry)

    return {
        "has_changes": graph_changed or files_changed or len(changed_files) > 0,
        "changed_files": changed_files,
        "graph_changed": graph_changed,
        "files_changed": files_changed,
    }


@app.post("/api/run-notebook")
def api_run_notebook(
    notebook_rel: str = Query(..., description="Notebook path relative to project root"),
    project_root: Optional[str] = Query(default=None),
):
    if os.environ.get("PIPELINE_VIZ_ALLOW_RUN", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "Notebook execution disabled. Set PIPELINE_VIZ_ALLOW_RUN=1")
    root = _resolve_root(project_root)
    nb = (root / notebook_rel).resolve()
    if not is_under_root(root, nb):
        raise HTTPException(400, "path escapes project root")
    if not nb.is_file() or nb.suffix.lower() != ".ipynb":
        raise HTTPException(400, "not a .ipynb file under project root")
    traced_inputs, traced_outputs = _run_notebook_execute_inplace(root, nb)
    if traced_inputs or traced_outputs:
        update_runtime_io(root, notebook_rel, traced_inputs, traced_outputs)
    _graph_cache.invalidate(root)
    return {
        "ok": True,
        "notebook": notebook_rel,
        "traced_io": {
            "inputs": sorted(traced_inputs),
            "outputs": sorted(traced_outputs),
        },
    }
