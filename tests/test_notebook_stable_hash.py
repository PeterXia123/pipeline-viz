"""Notebook 执行元数据不应影响稳定 hash。"""

import json
from pathlib import Path

from pipeline_viz.notebook_sanitize import sanitize_notebook_dict, stable_ipynb_sha256


def _minimal_nb() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {
                    "execution": {
                        "iopub.execute_input": "2026-03-27T01:52:39.092497Z",
                        "iopub.status.busy": "2026-03-27T01:52:39.092042Z",
                    }
                },
                "source": ["print(1)\n"],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {"ts": "2026-03-27T02:00:00Z"},
                        "data": {"text/plain": ["1"]},
                    }
                ],
            }
        ],
    }


def test_stable_ipynb_hash_ignores_execution_metadata(tmp_path: Path):
    a = _minimal_nb()
    b = json.loads(json.dumps(a))
    b["cells"][0]["execution_count"] = 99
    b["cells"][0]["metadata"]["execution"]["iopub.execute_input"] = "2099-01-01T00:00:00Z"
    b["cells"][0]["outputs"][0]["metadata"]["ts"] = "2099-01-01T00:00:00Z"

    pa = tmp_path / "a.ipynb"
    pb = tmp_path / "b.ipynb"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")

    assert stable_ipynb_sha256(pa) == stable_ipynb_sha256(pb)


def test_stable_ipynb_hash_ignores_metadata_language_info(tmp_path: Path):
    a = _minimal_nb()
    b = json.loads(json.dumps(a))
    b["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.9.6",
        "codemirror_mode": {"name": "ipython", "version": 3},
    }

    pa = tmp_path / "a.ipynb"
    pb = tmp_path / "b.ipynb"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")

    assert stable_ipynb_sha256(pa) == stable_ipynb_sha256(pb)


def test_stable_ipynb_hash_changes_when_code_changes(tmp_path: Path):
    a = _minimal_nb()
    b = json.loads(json.dumps(a))
    b["cells"][0]["source"] = ["print(2)\n"]

    pa = tmp_path / "a.ipynb"
    pb = tmp_path / "b.ipynb"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")

    assert stable_ipynb_sha256(pa) != stable_ipynb_sha256(pb)


def test_sanitize_notebook_dict_strips_volatile_fields():
    nb = _minimal_nb()
    sanitize_notebook_dict(nb)
    assert nb["cells"][0]["execution_count"] is None
    assert "execution" not in nb["cells"][0]["metadata"]
    assert nb["cells"][0]["outputs"][0].get("metadata") is None
