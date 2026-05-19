from __future__ import annotations

import difflib
import re
from pathlib import Path

from pipeline_viz.notebook_sanitize import ipynb_json_to_code_only_text


def read_text_safe(path: Path, max_bytes: int = 400_000) -> str:
    if not path.is_file():
        return ""
    try:
        if path.stat().st_size > max_bytes:
            return f"[文件过大，已省略 diff，>{max_bytes} bytes]\n"
    except OSError:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def unified_diff_text(rel_path: str, before: str, after: str) -> str:
    a = before.splitlines(True)
    b = after.splitlines(True)
    return "".join(
        difflib.unified_diff(
            a,
            b,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )


def diff_pair_for_display(rel_path: str, before: str, after: str) -> dict:
    ipynb = rel_path.lower().endswith(".ipynb")
    if ipynb:
        before = ipynb_json_to_code_only_text(before)
        after = ipynb_json_to_code_only_text(after)
    ud = unified_diff_text(rel_path, before, after)
    return {
        "unified": ud,
        "path": rel_path,
        "unchanged": before == after,
        "changed_lines": changed_lines_from_unified(ud),
        "diff_kind": "ipynb_code_only" if ipynb else "text",
    }


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def changed_lines_from_unified(unified: str) -> dict:
    """Extract changed line numbers from unified diff text."""
    old_lines: list[int] = []
    new_lines: list[int] = []
    old_ln = 0
    new_ln = 0

    for raw in unified.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            old_ln = int(m.group(1))
            new_ln = int(m.group(2))
            continue
        if raw.startswith("--- ") or raw.startswith("+++ "):
            continue
        if raw.startswith("-"):
            old_lines.append(old_ln)
            old_ln += 1
            continue
        if raw.startswith("+"):
            new_lines.append(new_ln)
            new_ln += 1
            continue
        if raw.startswith(" "):
            old_ln += 1
            new_ln += 1

    return {"old": old_lines, "new": new_lines}
