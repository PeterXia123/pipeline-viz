# Example: Cookiecutter-style data science layout

This example mimics a **Cookiecutter Data Science**–style repository: layered `data/`, `notebooks/`, `src/`, `models/`, `reports/`, and `references/`, wired for **pipeline-viz** (manifest + notebook parsing).

## Layout

| Path | Role |
|------|------|
| `data/raw/` | Immutable inputs |
| `data/interim/` | Intermediate merges |
| `data/processed/` | Model-ready tables |
| `data/external/` | Third-party reference data |
| `notebooks/` | Numbered analysis pipeline |
| `src/ccds_ds/` | Importable package (`from src.ccds_ds...` for viz graph edges) |
| `models/` | Serialized metrics / placeholders |
| `reports/` | Figures + exported metrics |
| `references/` | Papers, notes |
| `pipeline-manifest.yaml` | 仅补充解析器不易推断的边（本例主要为 `04_report`）；`geo_lookup` 由 `ingest.py` 内 `read_csv` 解析，不写在 manifest 里 |

## Use with pipeline-viz

Point **项目根目录** at this folder (or set `PIPELINE_VIZ_PROJECT_ROOT`), then load the graph. Edges follow reads/writes in notebooks and calls to `src/ccds_ds/*.py`.

## Run notebooks (optional)

```bash
cd example_cookiecutter_ds
pip install -r requirements.txt
# Run 01 → 02 → 03 → 04 in order from project root
```

Python imports use `from src.ccds_ds....` so the static import graph matches files under `src/ccds_ds/`.
