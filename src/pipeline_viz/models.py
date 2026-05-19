from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CodeEntry(BaseModel):
    path: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    project_root: str = "."
    codes: list[CodeEntry] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str
    kind: Literal["data", "notebook", "python"]
    path: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_code(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("kind") == "code":
            out = dict(data)
            p = str(out.get("path", ""))
            out["kind"] = "notebook" if p.endswith(".ipynb") else "python"
            return out
        return data


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    kind: Literal["input", "output", "notebook_call"]


class GraphPayload(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
