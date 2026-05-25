"""Edge-case tests for pipeline_viz.models module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_viz.models import (
    CodeEntry,
    GraphEdge,
    GraphNode,
    GraphPayload,
    Manifest,
)


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------

class TestGraphNode:
    def test_legacy_kind_code_with_ipynb_becomes_notebook(self):
        node = GraphNode.model_validate({
            "id": "code:nb.ipynb",
            "label": "nb.ipynb",
            "kind": "code",
            "path": "notebooks/nb.ipynb",
        })
        assert node.kind == "notebook"

    def test_legacy_kind_code_with_py_becomes_python(self):
        node = GraphNode.model_validate({
            "id": "code:script.py",
            "label": "script.py",
            "kind": "code",
            "path": "src/script.py",
        })
        assert node.kind == "python"

    def test_legacy_kind_code_with_no_extension_becomes_python(self):
        node = GraphNode.model_validate({
            "id": "code:Makefile",
            "label": "Makefile",
            "kind": "code",
            "path": "Makefile",
        })
        assert node.kind == "python"

    def test_valid_kind_data(self):
        node = GraphNode(id="data:raw.csv", label="raw.csv", kind="data", path="data/raw.csv")
        assert node.kind == "data"

    def test_valid_kind_notebook(self):
        node = GraphNode(id="code:nb.ipynb", label="nb.ipynb", kind="notebook", path="nb.ipynb")
        assert node.kind == "notebook"

    def test_valid_kind_python(self):
        node = GraphNode(id="code:m.py", label="m.py", kind="python", path="m.py")
        assert node.kind == "python"

    def test_invalid_kind_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphNode(id="x", label="x", kind="sql", path="x.sql")

    def test_invalid_kind_executable_raises(self):
        with pytest.raises(ValidationError):
            GraphNode(id="x", label="x", kind="executable", path="x.sh")

    def test_legacy_code_coercion_preserves_other_fields(self):
        node = GraphNode.model_validate({
            "id": "code:nb.ipynb",
            "label": "My Notebook",
            "kind": "code",
            "path": "nb.ipynb",
        })
        assert node.id == "code:nb.ipynb"
        assert node.label == "My Notebook"
        assert node.path == "nb.ipynb"
        assert node.kind == "notebook"


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------

class TestGraphEdge:
    def test_valid_kind_input(self):
        edge = GraphEdge(id="e1", source="a", target="b", kind="input")
        assert edge.kind == "input"

    def test_valid_kind_output(self):
        edge = GraphEdge(id="e2", source="a", target="b", kind="output")
        assert edge.kind == "output"

    def test_valid_kind_notebook_call(self):
        edge = GraphEdge(id="e3", source="a", target="b", kind="notebook_call")
        assert edge.kind == "notebook_call"

    def test_invalid_kind_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphEdge(id="e4", source="a", target="b", kind="import")

    def test_another_invalid_kind(self):
        with pytest.raises(ValidationError):
            GraphEdge(id="e5", source="a", target="b", kind="reference")


# ---------------------------------------------------------------------------
# GraphPayload
# ---------------------------------------------------------------------------

class TestGraphPayload:
    def test_empty_nodes_and_edges(self):
        payload = GraphPayload(nodes=[], edges=[])
        assert payload.nodes == []
        assert payload.edges == []

    def test_payload_with_one_node_and_one_edge(self):
        n = GraphNode(id="code:a.py", label="a.py", kind="python", path="a.py")
        e = GraphEdge(id="e1", source="code:a.py", target="data:x.csv", kind="output")
        payload = GraphPayload(nodes=[n], edges=[e])
        assert len(payload.nodes) == 1
        assert len(payload.edges) == 1


# ---------------------------------------------------------------------------
# CodeEntry
# ---------------------------------------------------------------------------

class TestCodeEntry:
    def test_basic_construction(self):
        entry = CodeEntry(path="notebooks/etl.ipynb")
        assert entry.path == "notebooks/etl.ipynb"
        assert entry.inputs == []
        assert entry.outputs == []

    def test_with_inputs_and_outputs(self):
        entry = CodeEntry(
            path="etl.ipynb",
            inputs=["data/raw.csv"],
            outputs=["data/clean.csv", "data/summary.csv"],
        )
        assert len(entry.inputs) == 1
        assert len(entry.outputs) == 2


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_default_project_root(self):
        m = Manifest()
        assert m.project_root == "."
        assert m.codes == []

    def test_with_codes(self):
        m = Manifest(
            project_root="/tmp/proj",
            codes=[CodeEntry(path="nb.ipynb", inputs=["a.csv"])],
        )
        assert len(m.codes) == 1
        assert m.codes[0].path == "nb.ipynb"
