"""Application controller.

Owns run lifecycle - start, pause, resume, stop, reset - and drives the engine
on a worker thread so neither the API nor the desktop UI ever blocks on the
simulation loop.  Simulation rate is decoupled from display rate: the loop can
run flat out while the UI polls at a human refresh rate.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..core.types import RunState
from .engine import SmartScanEngine

logger = logging.getLogger(__name__)


class SimulationController:
    """Single-run controller shared by the API and the desktop client."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._engine: Optional[SmartScanEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._speed_hz: float = 0.0          # 0 = run as fast as the CPU allows
        self._max_slots: Optional[int] = None
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self.last_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ access
    @property
    def engine(self) -> Optional[SmartScanEngine]:
        return self._engine

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def add_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for callback in list(self._listeners):
            try:
                callback(event, payload)
            except Exception:  # pragma: no cover - a bad listener must not stop a run
                logger.exception("listener failed for event %s", event)

    # ------------------------------------------------------------------ control
    def start(
        self,
        strategy: Optional[str] = None,
        scenario: Optional[str] = None,
        seed: Optional[int] = None,
        adaptive: Optional[bool] = None,
        speed: float = 0.0,
        max_slots: Optional[int] = None,
        persist: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a fresh run and begin executing it."""
        with self._lock:
            if self.is_running:
                self.stop(join=True)
            cfg = self.cfg.merged(overrides) if overrides else self.cfg
            self._engine = SmartScanEngine(
                cfg=cfg,
                strategy=strategy,
                seed=seed,
                scenario=scenario,
                adaptive_enabled=adaptive,
                persist=persist,
            )
            self._speed_hz = float(max(0.0, speed))
            self._max_slots = max_slots
            self._stop_event.clear()
            self._pause_event.clear()
            self._engine.state = RunState.RUNNING
            self._thread = threading.Thread(target=self._loop, name="smartscan-run", daemon=True)
            self._thread.start()
            self._emit("run_started", {"run_id": self._engine.run_id})
            return self._engine.status()

    def _loop(self) -> None:
        engine = self._engine
        assert engine is not None
        interval = 1.0 / self._speed_hz if self._speed_hz > 0 else 0.0
        executed = 0
        next_tick = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.05)
                    next_tick = time.perf_counter()
                    continue
                if self._max_slots is not None and executed >= self._max_slots:
                    break
                with self._lock:
                    step = engine.step()
                if step is None:
                    break
                executed += 1
                if interval > 0.0:
                    next_tick += interval
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.perf_counter()
                elif executed % 256 == 0:
                    time.sleep(0)          # keep the GIL fair to the API thread
        except Exception as exc:  # pragma: no cover - surfaced through status
            logger.exception("simulation loop failed")
            with self._lock:
                engine.error = str(exc)
                engine.state = RunState.ERROR
            self._emit("run_error", {"run_id": engine.run_id, "error": str(exc)})
            return
        finally:
            with self._lock:
                if engine.state not in (RunState.ERROR, RunState.STOPPED):
                    engine.finish()
                self.last_summary = engine.summary()
        self._emit("run_finished", {"run_id": engine.run_id, "summary": self.last_summary})

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            if self._engine is not None:
                self._pause_event.set()
                self._engine.pause()
            return self.status()

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            if self._engine is not None:
                self._pause_event.clear()
                self._engine.resume()
            return self.status()

    def stop(self, join: bool = False) -> Dict[str, Any]:
        thread = self._thread
        self._stop_event.set()
        self._pause_event.clear()
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            if self._engine is not None:
                self._engine.stop()
                self.last_summary = self._engine.summary()
            return self.status()

    def reset(self) -> Dict[str, Any]:
        self.stop(join=True)
        with self._lock:
            self._engine = None
            self._thread = None
            self.last_summary = None
        return {"state": RunState.IDLE.value}

    def set_speed(self, speed: float) -> Dict[str, Any]:
        with self._lock:
            self._speed_hz = float(max(0.0, speed))
        return {"speed": self._speed_hz}

    # -------------------------------------------------------------------- read
    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._engine is None:
                return {"state": RunState.IDLE.value, "run_id": None, "slot": 0,
                        "speed": self._speed_hz}
            status = self._engine.status()
            status["speed"] = self._speed_hz
            status["thread_alive"] = self.is_running
            return status

    def snapshot(self, include_grid: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self._engine is None:
                return {"state": RunState.IDLE.value}
            return self._engine.state_snapshot(include_grid=include_grid)

    def activity_window(self, width: int = 200) -> Dict[str, Any]:
        with self._lock:
            if self._engine is None:
                return {"grid": [], "scans": [], "n_bands": 0, "start_slot": 0, "end_slot": 0}
            return self._engine.activity_window(width=width)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            if self._engine is None:
                return self.last_summary or {}
            return self._engine.summary()

    def adaptive_status(self) -> Dict[str, Any]:
        with self._lock:
            if self._engine is None:
                return {"enabled": False, "state": "IDLE"}
            return self._engine.adaptive.status()


__all__ = ["SimulationController"]
