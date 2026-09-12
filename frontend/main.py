"""Desktop entry point.

Starts the local FastAPI service on a background thread, waits for it to answer
its health probe, then opens the workstation window.  Everything runs on this
machine: no cloud service and no internet connection is required.

    python frontend/main.py                # embedded service
    python frontend/main.py --no-server    # attach to a service already running
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from backend.app.config import load_config  # noqa: E402
from frontend import theme  # noqa: E402
from frontend.api_client import ApiClient  # noqa: E402
from frontend.main_window import MainWindow  # noqa: E402


def start_service(host: str, port: int) -> threading.Thread:
    """Run uvicorn in a daemon thread so the GUI owns the main thread."""
    import uvicorn

    from backend.app.main import app

    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="smartscan-api", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smart Scan Strategy workstation")
    parser.add_argument("--no-server", action="store_true",
                        help="Do not start the embedded service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    host = args.host or str(cfg.get_path("api.host", "127.0.0.1"))
    port = int(args.port or cfg.get_path("api.port", 8077))

    if not args.no_server:
        start_service(host, port)

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Smart Scan Strategy")
    app.setOrganizationName("Smart Scan Research")
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    app.setFont(QFont("Segoe UI", 9))

    client = ApiClient(f"http://{host}:{port}")
    if not client.wait_until_ready(timeout=40.0):
        QMessageBox.critical(
            None, "Smart Scan Strategy",
            f"The local service at http://{host}:{port} did not start.\n\n"
            "Start it manually with:  python scripts/run_api.py")
        return 1

    window = MainWindow(client)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
