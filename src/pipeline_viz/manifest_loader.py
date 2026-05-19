from __future__ import annotations

from pathlib import Path

import yaml

from pipeline_viz.models import CodeEntry, Manifest


def load_manifest(path: Path) -> Manifest:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return Manifest.model_validate(data)


def manifest_by_notebook(manifest: Manifest) -> dict[str, CodeEntry]:
    out: dict[str, CodeEntry] = {}
    for c in manifest.codes:
        rel = c.path.replace("\\", "/").strip()
        out[rel] = c
    return out
