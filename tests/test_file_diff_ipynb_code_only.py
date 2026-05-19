"""ipynb 的 diff 与变更检测仅基于 code cell，不包含 JSON metadata。"""

from __future__ import annotations

import json

from pipeline_viz.file_diff import diff_pair_for_display


def test_diff_pair_ipynb_ignores_metadata_diff():
    nb_a = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"language_info": {"name": "python", "version": "3.9"}},
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["x = 1\n"], "outputs": []},
        ],
    }
    nb_b = json.loads(json.dumps(nb_a))
    nb_b["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.9.6",
        "pygments_lexer": "ipython3",
    }

    d = diff_pair_for_display(
        "n.ipynb",
        json.dumps(nb_a),
        json.dumps(nb_b),
    )
    assert d["diff_kind"] == "ipynb_code_only"
    assert d["unchanged"] is True
    assert d["unified"] == ""


def test_diff_pair_ipynb_shows_code_change():
    nb_a = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["x = 1\n"], "outputs": []},
        ],
    }
    nb_b = json.loads(json.dumps(nb_a))
    nb_b["cells"][0]["source"] = ["x = 2\n"]

    d = diff_pair_for_display(
        "n.ipynb",
        json.dumps(nb_a),
        json.dumps(nb_b),
    )
    assert d["unchanged"] is False
    assert "x = 1" in d["unified"] or "-x" in d["unified"]
    assert "x = 2" in d["unified"] or "+x" in d["unified"]
    assert d["diff_kind"] == "ipynb_code_only"
