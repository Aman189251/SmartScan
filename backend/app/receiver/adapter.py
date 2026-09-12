"""Receiver adapter boundary.

Everything above this file - features, ML, Bayesian belief, scheduling,
evaluation, database, UI - is hardware independent.  Only an adapter knows how
to talk to a particular source of observations, and every adapter returns the
same :class:`NormalizedObservation`.

Adapters shipped here:

* ``SimulatorReceiverAdapter``  - the synthetic environment, used for development,
  demonstration and every experiment in this repository.
* ``RecordedDataAdapter``       - deterministic replay of a stored observation log.
* ``ExternalReceiverAdapter``   - reference client for an external receiver service
  that speaks the documented line-delimited JSON protocol in ``docs/INTEGRATION.md``.

An integration with a specific external or laboratory receiver is implemented by
writing one more subclass.  No other module changes.  The wire format used by
``ExternalReceiverAdapter`` is this project's own convention: it must be replaced
with the approved interface specification before any real integration, and no
claim of compatibility with an undisclosed interface is made here.
"""

from __future__ import annotations

import json
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.types import NormalizedObservation, Outcome, ScanCommand


class ReceiverAdapter(ABC):
    """Abstract interface between the Smart Scan core and any observation source."""

    #: Identifies the source in the normalised observation stream.
    source_id: str = "adapter"

    # -- lifecycle --------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Open the underlying resource. Must be idempotent."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the underlying resource. Must be safe to call twice."""

    # -- capability -------------------------------------------------------
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Report band count, capacity, timing limits and supported features."""

    @abstractmethod
    def configure_scan(self, **params: Any) -> None:
        """Apply scan configuration before a run starts."""

    # -- execution --------------------------------------------------------
    @abstractmethod
    def start_scan(self, command: ScanCommand) -> None:
        """Dispatch one scan instruction."""

    @abstractmethod
    def stop_scan(self) -> None:
        """Abort any scan in progress."""

    @abstractmethod
    def get_observation(self) -> NormalizedObservation:
        """Return the normalised observation for the dispatched scan."""

    # -- health -----------------------------------------------------------
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Current adapter/receiver state."""

    def health_check(self) -> Dict[str, Any]:
        """Default health probe: status plus a reachability flag."""
        try:
            status = self.get_status()
            return {"healthy": True, "source_id": self.source_id, "status": status}
        except Exception as exc:  # pragma: no cover - defensive
            return {"healthy": False, "source_id": self.source_id, "error": str(exc)}

    # -- convenience ------------------------------------------------------
    def execute(self, command: ScanCommand) -> NormalizedObservation:
        """Dispatch a scan and collect its observation."""
        self.start_scan(command)
        return self.get_observation()

    def __enter__(self) -> "ReceiverAdapter":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()


class SimulatorReceiverAdapter(ReceiverAdapter):
    """Wraps the synthetic environment plus receiver simulator."""

    source_id = "simulator"

    def __init__(self, receiver, run_id: str = "run"):
        self.receiver = receiver
        self.run_id = run_id
        self._connected = False
        self._pending: Optional[ScanCommand] = None
        self._last: Optional[NormalizedObservation] = None

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        caps = self.receiver.constraints.to_dict()
        caps.update({"source": self.source_id, "supports_replay": False, "simulated": True})
        return caps

    def configure_scan(self, **params: Any) -> None:
        if "pd" in params:
            self.receiver.pd = float(params["pd"])
        if "pfa" in params:
            self.receiver.pfa = float(params["pfa"])
        if "run_id" in params:
            self.run_id = str(params["run_id"])

    def start_scan(self, command: ScanCommand) -> None:
        if not self._connected:
            raise RuntimeError("adapter not connected")
        self._pending = command

    def stop_scan(self) -> None:
        self._pending = None

    def get_observation(self) -> NormalizedObservation:
        if self._pending is None:
            raise RuntimeError("no scan dispatched")
        self._last = self.receiver.observe(self._pending, run_id=self.run_id, source_id=self.source_id)
        self._pending = None
        return self._last

    def get_status(self) -> Dict[str, Any]:
        status = self.receiver.status.to_dict()
        status["connected"] = self._connected
        return status


class RecordedDataAdapter(ReceiverAdapter):
    """Replays a stored observation log (JSON lines) for repeatable research runs."""

    source_id = "recorded"

    def __init__(self, path: str | Path, run_id: str = "replay", loop: bool = False):
        self.path = Path(path)
        self.run_id = run_id
        self.loop = loop
        self._records: List[Dict[str, Any]] = []
        self._cursor = 0
        self._pending: Optional[ScanCommand] = None
        self._connected = False

    def connect(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"replay source not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as fh:
            self._records = [json.loads(line) for line in fh if line.strip()]
        self._cursor = 0
        self._connected = True

    def disconnect(self) -> None:
        self._records.clear()
        self._connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        bands = {int(r["band_id"]) for r in self._records} or {0}
        return {
            "source": self.source_id,
            "n_bands": max(bands) + 1,
            "capacity": 1,
            "scan_duration": 1,
            "records": len(self._records),
            "supports_replay": True,
            "simulated": False,
        }

    def configure_scan(self, **params: Any) -> None:
        if "run_id" in params:
            self.run_id = str(params["run_id"])
        if "loop" in params:
            self.loop = bool(params["loop"])

    def start_scan(self, command: ScanCommand) -> None:
        if not self._connected:
            raise RuntimeError("adapter not connected")
        self._pending = command

    def stop_scan(self) -> None:
        self._pending = None

    def get_observation(self) -> NormalizedObservation:
        if self._pending is None:
            raise RuntimeError("no scan dispatched")
        if self._cursor >= len(self._records):
            if not self.loop:
                raise StopIteration("replay exhausted")
            self._cursor = 0
        record = self._records[self._cursor]
        self._cursor += 1
        cmd, self._pending = self._pending, None
        outcome = Outcome(record["outcome"])
        return NormalizedObservation(
            run_id=self.run_id,
            source_id=self.source_id,
            slot=cmd.slot,
            timestamp=float(record.get("timestamp", time.time())),
            band_id=int(record.get("band_id", cmd.band_id)),
            scan_duration=int(record.get("scan_duration", 1)),
            outcome=outcome,
            detected=outcome.is_detection,
            quality=float(record.get("quality", 1.0)),
            receiver_state=str(record.get("receiver_state", "REPLAY")),
            metadata={"replay_index": self._cursor - 1},
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "cursor": self._cursor,
            "records": len(self._records),
            "state": "REPLAY",
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)


class ExternalReceiverAdapter(ReceiverAdapter):
    """Reference client for an external receiver service.

    Protocol (documented in ``docs/INTEGRATION.md``): line-delimited JSON over
    TCP.  The core sends a request object and reads exactly one response line.

        -> {"cmd": "capabilities"}
        <- {"n_bands": 32, "capacity": 1, "scan_duration": 1}

        -> {"cmd": "scan", "run_id": "...", "slot": 41, "band_id": 7, "dwell": 1}
        <- {"outcome": "HIT", "quality": 0.93, "receiver_state": "READY"}

    ``outcome`` may be reported either as one of HIT / MISS / FALSE_ALARM /
    CORRECT_NON_DETECTION, or as a bare ``detected`` boolean, in which case a
    detection maps to HIT and a non-detection to CORRECT_NON_DETECTION.  A live
    source has no ground truth, so MISS and FALSE_ALARM are only distinguishable
    when the external service supplies them.

    This wire format is a placeholder convention chosen so the integration path
    can be exercised end to end.  Replace it with the approved interface
    specification when one is available; nothing outside this class changes.
    """

    source_id = "external"

    def __init__(self, host: str = "127.0.0.1", port: int = 9500, timeout: float = 5.0,
                 run_id: str = "external"):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.run_id = run_id
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        self._pending: Optional[ScanCommand] = None
        self._capabilities: Dict[str, Any] = {}

    # -- transport --------------------------------------------------------
    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("adapter not connected")
        self._sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        while b"\n" not in self._buffer:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("external receiver closed the connection")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._buffer = b""
        self._capabilities = self._send({"cmd": "capabilities"})

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._buffer = b""

    # -- interface --------------------------------------------------------
    def get_capabilities(self) -> Dict[str, Any]:
        caps = dict(self._capabilities)
        caps.setdefault("n_bands", 32)
        caps.setdefault("capacity", 1)
        caps.setdefault("scan_duration", 1)
        caps["source"] = self.source_id
        caps["simulated"] = False
        return caps

    def configure_scan(self, **params: Any) -> None:
        if "run_id" in params:
            self.run_id = str(params["run_id"])
        payload = {k: v for k, v in params.items() if k != "run_id"}
        if payload:
            self._send({"cmd": "configure", "params": payload})

    def start_scan(self, command: ScanCommand) -> None:
        self._pending = command

    def stop_scan(self) -> None:
        self._pending = None
        if self._sock is not None:
            self._send({"cmd": "stop"})

    def get_observation(self) -> NormalizedObservation:
        if self._pending is None:
            raise RuntimeError("no scan dispatched")
        cmd, self._pending = self._pending, None
        response = self._send({
            "cmd": "scan",
            "run_id": self.run_id,
            "slot": cmd.slot,
            "band_id": cmd.band_id,
            "dwell": cmd.dwell,
        })
        if "outcome" in response:
            outcome = Outcome(str(response["outcome"]).upper())
        else:
            outcome = Outcome.HIT if response.get("detected") else Outcome.CORRECT_NON_DETECTION
        return NormalizedObservation(
            run_id=self.run_id,
            source_id=self.source_id,
            slot=cmd.slot,
            timestamp=float(response.get("timestamp", time.time())),
            band_id=int(response.get("band_id", cmd.band_id)),
            scan_duration=int(response.get("scan_duration", cmd.dwell)),
            outcome=outcome,
            detected=outcome.is_detection,
            quality=float(response.get("quality", 1.0)),
            receiver_state=str(response.get("receiver_state", "READY")),
            metadata={"external": True},
        )

    def get_status(self) -> Dict[str, Any]:
        if self._sock is None:
            return {"connected": False, "state": "DISCONNECTED"}
        try:
            status = self._send({"cmd": "status"})
        except Exception as exc:  # pragma: no cover - network dependent
            return {"connected": False, "state": "ERROR", "error": str(exc)}
        status["connected"] = True
        return status


ADAPTER_REGISTRY = {
    "simulator": SimulatorReceiverAdapter,
    "recorded": RecordedDataAdapter,
    "external": ExternalReceiverAdapter,
}

__all__ = [
    "ReceiverAdapter",
    "SimulatorReceiverAdapter",
    "RecordedDataAdapter",
    "ExternalReceiverAdapter",
    "ADAPTER_REGISTRY",
]
