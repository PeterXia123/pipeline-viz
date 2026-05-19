"""Normalize .ipynb JSON so run timestamps / execution metadata do not churn hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def ipynb_code_fingerprint_bytes(nb: dict[str, Any]) -> bytes:
    """
    仅由 code cell 的源码拼接而成（按出现顺序），用于 hash 与 diff。
    忽略：nbformat metadata、markdown cell、outputs、execution_count 等。
    """
    parts: list[str] = []
    i = 0
    for cell in nb.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        parts.append(f"# --- code cell {i} ---\n{src}")
        i += 1
    return "\n".join(parts).encode("utf-8")


def ipynb_json_to_code_only_text(txt: str) -> str:
    """将 .ipynb 原始 JSON 文本转为仅含 code cell 的可读文本；解析失败则原样返回。"""
    if not (txt and txt.strip()):
        return ""
    try:
        nb = json.loads(txt)
        if not isinstance(nb, dict):
            return txt
        return ipynb_code_fingerprint_bytes(nb).decode("utf-8")
    except Exception:
        return txt


def sanitize_notebook_dict(notebook: dict[str, Any]) -> dict[str, Any]:
    """清理 notebook JSON：去掉执行计数、时间戳类 metadata（含 outputs 内）。"""
    nm = notebook.get("metadata")
    if isinstance(nm, dict):
        nm.pop("execution", None)

    for cell in notebook.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cmd = cell.get("metadata")
            if isinstance(cmd, dict):
                cmd.pop("execution", None)
            for out in cell.get("outputs") or []:
                if isinstance(out, dict):
                    out.pop("metadata", None)
    return notebook


def stable_ipynb_sha256(path: Path) -> str:
    """
    对 .ipynb 仅按 **code cell 源码** 做 SHA256。
    同一套代码在仅改 metadata / outputs / Jupyter 保存格式时应 hash 不变。
    解析失败时回退为原始文件字节 hash。
    """
    try:
        with path.open(encoding="utf-8") as f:
            nb = json.load(f)
        if not isinstance(nb, dict):
            return raw_file_sha256(path)
        body = ipynb_code_fingerprint_bytes(nb)
        return hashlib.sha256(body).hexdigest()
    except Exception:
        return raw_file_sha256(path)


def raw_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
