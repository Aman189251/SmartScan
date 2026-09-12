"""Thin HTTP client for the local Smart Scan service.

The desktop UI never imports the engine directly.  It goes through this client,
which keeps the interface boundary real: the same calls work against an
in-process service thread or a separately launched one.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8077", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------- transport
    def _get(self, path: str, **params: Any) -> Dict[str, Any]:
        try:
            response = self._client.get(path, params=params or None)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise ApiError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = self._client.post(path, json=payload or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise ApiError(f"POST {path} failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until the service answers /health, or give up."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._client.get("/health", timeout=1.0).raise_for_status()
                return True
            except httpx.HTTPError:
                time.sleep(0.2)
        return False

    # ----------------------------------------------------------- run control
    def start(self, **kwargs: Any) -> Dict[str, Any]:
        return self._post("/simulation/start", kwargs)

    def pause(self) -> Dict[str, Any]:
        return self._post("/simulation/pause")

    def resume(self) -> Dict[str, Any]:
        return self._post("/simulation/resume")

    def stop(self) -> Dict[str, Any]:
        return self._post("/simulation/stop")

    def reset(self) -> Dict[str, Any]:
        return self._post("/simulation/reset")

    def set_speed(self, speed: float) -> Dict[str, Any]:
        return self._post("/simulation/speed", {"speed": speed})

    # ------------------------------------------------------------------ read
    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def status(self) -> Dict[str, Any]:
        return self._get("/simulation/state")

    def snapshot(self, include_grid: bool = True) -> Dict[str, Any]:
        return self._get("/simulation/snapshot", include_grid=include_grid)

    def activity(self, width: int = 200) -> Dict[str, Any]:
        return self._get("/simulation/activity", width=width)

    def scenarios(self) -> List[Dict[str, Any]]:
        return self._get("/simulation/scenarios").get("scenarios", [])

    def strategies(self) -> List[Dict[str, Any]]:
        return self._get("/simulation/strategies").get("strategies", [])

    def adaptive(self) -> Dict[str, Any]:
        return self._get("/adaptive")

    def metrics(self) -> Dict[str, Any]:
        return self._get("/metrics")

    def runs(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self._get("/simulation/runs", limit=limit).get("runs", [])

    def run_detail(self, run_id: str) -> Dict[str, Any]:
        return self._get(f"/simulation/runs/{run_id}")

    # ---------------------------------------------------------------- models
    def models(self) -> Dict[str, Any]:
        return self._get("/models")

    def model_detail(self, model_id: str) -> Dict[str, Any]:
        return self._get(f"/models/{model_id}")

    def promote_model(self, model_id: str) -> Dict[str, Any]:
        return self._post(f"/models/{model_id}/promote")

    def train(self, **kwargs: Any) -> Dict[str, Any]:
        return self._post("/models/train", kwargs)

    # ----------------------------------------------------------- experiments
    def run_experiment(self, **kwargs: Any) -> Dict[str, Any]:
        return self._post("/experiments/run", kwargs)

    def experiments(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self._get("/experiments", limit=limit).get("experiments", [])

    def experiment_files(self) -> List[Dict[str, Any]]:
        return self._get("/experiments/files").get("files", [])

    def experiment_file(self, name: str) -> Dict[str, Any]:
        return self._get(f"/experiments/file/{name}")

    # ------------------------------------------------------------------ jobs
    def jobs(self) -> List[Dict[str, Any]]:
        return self._get("/jobs").get("jobs", [])

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._get(f"/jobs/{job_id}")


__all__ = ["ApiClient", "ApiError"]
