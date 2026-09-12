"""FastAPI application: the local service boundary.

This is a boundary, not a web product.  The desktop client is the interface; the
API exists so the UI never reaches into engine internals, so the engine can be
driven headlessly by scripts and tests, and so a different front end could be
attached without touching the intelligence core.  It binds to loopback and needs
no internet connection.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI

from .api import experiments as experiments_routes
from .api import metrics as metrics_routes
from .api import scheduler as scheduler_routes
from .api import simulation as simulation_routes
from .api.schemas import StartRequest
from .api.state import AppState, get_state
from .config import LOG_DIR, load_config

logger = logging.getLogger("smartscan")

API_TITLE = "Smart Scan Strategy"
API_VERSION = "1.0.0"


def configure_logging(cfg) -> None:
    level = getattr(logging, str(cfg.get_path("logging.level", "INFO")).upper(), logging.INFO)
    log_file = LOG_DIR / str(cfg.get_path("logging.file", "smartscan.log"))
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except OSError:  # pragma: no cover - read-only install
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState.instance()
    configure_logging(state.cfg)
    logger.info("Smart Scan service ready (database initialised, registry loaded)")
    yield
    try:
        state.controller.stop(join=True)
    except Exception:  # pragma: no cover
        logger.exception("failed to stop the controller cleanly")


def create_app(cfg=None) -> FastAPI:
    if cfg is not None:
        AppState.reset_instance()
        AppState.instance(cfg)

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Local service layer for the Smart Scan Strategy research workstation. "
            "Synthetic environment, abstract bands, simulated observations."
        ),
        lifespan=lifespan,
    )

    app.include_router(simulation_routes.router)
    app.include_router(scheduler_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(experiments_routes.router)

    # -- short aliases kept for convenience and for the roadmap's endpoint list
    @app.get("/health", tags=["system"])
    def health(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        engine = state.controller.engine
        adapter_health = engine.adapter.health_check() if engine is not None else {
            "healthy": True, "source_id": "none", "status": "idle"}
        return {
            "status": "ok",
            "version": API_VERSION,
            "run_state": state.controller.status().get("state"),
            "adapter": adapter_health,
            "database": str(state.cfg.get_path("database.url", "")),
        }

    @app.get("/status", tags=["system"])
    def status(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        return state.controller.status()

    @app.post("/start", tags=["system"])
    def start(request: StartRequest, state: AppState = Depends(get_state)) -> Dict[str, Any]:
        return simulation_routes.start(request, state)

    @app.post("/pause", tags=["system"])
    def pause(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        return state.controller.pause()

    @app.post("/resume", tags=["system"])
    def resume(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        return state.controller.resume()

    @app.post("/stop", tags=["system"])
    def stop(state: AppState = Depends(get_state)) -> Dict[str, Any]:
        return state.controller.stop(join=True)

    @app.get("/", tags=["system"])
    def root() -> Dict[str, Any]:
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "scope": ("Synthetic receiver-scheduling research simulator. Abstract bands, "
                      "simulated observations, no operational signal parameters."),
        }

    return app


app = create_app()


def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    """Run the service with uvicorn (used by scripts and the desktop launcher)."""
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "backend.app.main:app" if reload else app,
        host=host or str(cfg.get_path("api.host", "127.0.0.1")),
        port=int(port or cfg.get_path("api.port", 8077)),
        reload=reload,
        log_level=str(cfg.get_path("logging.level", "info")).lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    serve()
