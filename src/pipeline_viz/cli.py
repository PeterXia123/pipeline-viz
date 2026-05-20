from __future__ import annotations

import argparse
import os


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="pipeline-viz")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--allow-run", action="store_true", help="允许通过 API 执行 notebook")
    args = p.parse_args()
    os.environ.setdefault("PIPELINE_VIZ_PROJECT_ROOT", os.getcwd())
    if args.allow_run:
        os.environ["PIPELINE_VIZ_ALLOW_RUN"] = "1"
    uvicorn.run(
        "pipeline_viz.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
