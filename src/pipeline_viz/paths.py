from __future__ import annotations

import os
import re
from pathlib import Path

# 扫描 `*.ipynb` / `*.py` 时：任一一层目录名为下列之一则跳过（如 Jupyter 检查点目录）。
SCAN_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        "node_modules",
        ".pipeline-viz",
        ".ipynb_checkpoints",
    }
)


def rel_has_scan_skip_dir(rel: str) -> bool:
    """相对路径（POSIX）是否经过应排除的目录段。"""
    if not rel:
        return False
    parts = rel.replace("\\", "/").strip("/").split("/")
    return any(p in SCAN_SKIP_DIR_NAMES for p in parts)


def normalize_rel(project_root: Path, raw: str) -> str:
    """Return POSIX-like path relative to project root."""
    s = raw.strip().strip('"').strip("'")
    if not s or s.startswith("#"):
        return ""
    p = Path(s)
    if p.is_absolute():
        try:
            p = p.relative_to(project_root.resolve())
        except ValueError:
            p = Path(os.path.relpath(str(p), str(project_root.resolve())))
    parts = [x for x in p.parts if x not in ("..", ".")]
    return "/".join(parts).replace("\\", "/")


def is_under_root(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


DATA_EXT = re.compile(
    r"\.(csv|tsv|parquet|pq|json|jsonl|ndjson|feather|ftr|pkl|pickle|"
    r"joblib|h5|hdf5|txt|xml|yaml|yml|npz|npy|xlsx|xls|orc|ipc|arrow)$",
    re.IGNORECASE,
)


def data_id(rel: str) -> str:
    return f"data:{rel}"


def code_id(rel: str) -> str:
    return f"code:{rel}"
