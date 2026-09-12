"""Start the local FastAPI service on its own.

    python scripts/run_api.py --port 8077

Useful for headless work, for API tests, or when the desktop client should
attach to an already-running service via ``python frontend/main.py --no-server``.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from backend.app.config import load_config
from backend.app.main import serve


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run the Smart Scan local service")
    parser.add_argument("--host", default=str(cfg.get_path("api.host", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(cfg.get_path("api.port", 8077)))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"Smart Scan service on http://{args.host}:{args.port}  (docs at /docs)")
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
