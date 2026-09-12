"""Shared application state for the local service layer.

One controller, one config, one model registry per process.  Long jobs -
dataset generation, training, benchmarks - run on worker threads and are
tracked here so the UI can poll their progress instead of blocking on a request.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import Config, load_config
from ..core.controller import SimulationController
from ..db.database import init_database, sync_model_registry
from ..ml.model_registry import ModelRegistry


class Job:
    """A background task with progress the UI can poll."""

    def __init__(self, job_id: str, kind: str, detail: str = ""):
        self.id = job_id
        self.kind = kind
        self.detail = detail
        self.status = "PENDING"
        self.progress = 0.0
        self.message = ""
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.finished_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "detail": self.detail,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, target: Callable[[Job], Dict[str, Any]], detail: str = "") -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, detail)
        with self._lock:
            self._jobs[job.id] = job

        def runner() -> None:
            job.status = "RUNNING"
            try:
                job.result = target(job)
                job.status = "FINISHED"
                job.progress = 1.0
            except Exception as exc:
                job.status = "ERROR"
                job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
            finally:
                job.finished_at = datetime.now(timezone.utc)

        threading.Thread(target=runner, name=f"job-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def active(self, kind: Optional[str] = None) -> Optional[Job]:
        with self._lock:
            for job in self._jobs.values():
                if job.status in ("PENDING", "RUNNING") and (kind is None or job.kind == kind):
                    return job
        return None


class AppState:
    """Process-wide singleton wired at application startup."""

    _instance: Optional["AppState"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        init_database(self.cfg)
        self.registry = ModelRegistry()
        self.controller = SimulationController(self.cfg)
        self.jobs = JobManager()
        try:
            sync_model_registry(self.cfg, self.registry)
        except Exception:  # pragma: no cover - registry mirror is advisory
            pass

    @classmethod
    def instance(cls, cfg: Optional[Config] = None) -> "AppState":
        with cls._lock:
            if cls._instance is None:
                cls._instance = AppState(cfg)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None


def get_state() -> AppState:
    """FastAPI dependency."""
    return AppState.instance()


__all__ = ["AppState", "JobManager", "Job", "get_state"]
