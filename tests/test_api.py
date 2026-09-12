"""Local service layer: endpoints, run lifecycle and the adapter boundary."""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.app.api.state import AppState
from backend.app.core.engine import SmartScanEngine
from backend.app.core.types import Outcome, ScanCommand
from backend.app.main import create_app
from backend.app.receiver.adapter import (
    ADAPTER_REGISTRY,
    ExternalReceiverAdapter,
    RecordedDataAdapter,
    SimulatorReceiverAdapter,
)
from backend.app.simulator.receiver import ReceiverSimulator


@pytest.fixture
def client(cfg):
    AppState.reset_instance()
    app = create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client
    AppState.reset_instance()


def test_health_and_root(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert "adapter" in health
    assert client.get("/").json()["name"]


def test_catalogues_are_served(client):
    scenarios = client.get("/simulation/scenarios").json()["scenarios"]
    strategies = client.get("/simulation/strategies").json()["strategies"]
    assert {s["key"] for s in scenarios} >= {"mixed", "shift", "burst"}
    assert {s["key"] for s in strategies} >= {"random", "smart_scan", "ucb"}


def test_run_lifecycle(client):
    started = client.post("/simulation/start", json={
        "strategy": "smart_scan", "scenario": "mixed", "seed": 5,
        "speed": 0, "max_slots": 60, "persist": True,
    }).json()
    assert started["state"] in ("RUNNING", "FINISHED")
    run_id = started["run_id"]

    deadline = time.time() + 20
    while time.time() < deadline:
        status = client.get("/simulation/state").json()
        if status.get("slot", 0) >= 40 or status.get("state") in ("FINISHED", "STOPPED"):
            break
        time.sleep(0.1)

    status = client.get("/simulation/state").json()
    assert status["run_id"] == run_id
    assert status["slot"] > 0

    client.post("/simulation/stop")
    assert client.get("/simulation/state").json()["state"] in ("STOPPED", "FINISHED")

    detail = client.get(f"/simulation/runs/{run_id}").json()
    assert detail["run_id"] == run_id
    assert client.get("/simulation/runs").json()["runs"]


def test_live_state_endpoints_expose_predictions_and_beliefs(client):
    client.post("/simulation/start", json={"seed": 3, "speed": 0, "max_slots": 80})
    deadline = time.time() + 20
    while time.time() < deadline:
        if client.get("/simulation/state").json().get("slot", 0) > 30:
            break
        time.sleep(0.1)

    predictions = client.get("/predictions").json()
    beliefs = client.get("/beliefs").json()
    scheduler = client.get("/scheduler/state").json()
    adaptive = client.get("/adaptive").json()
    activity = client.get("/simulation/activity", params={"width": 50}).json()

    assert len(predictions["probabilities"]) == len(beliefs["posterior_means"]) > 0
    assert all(0.0 <= p <= 1.0 for p in predictions["probabilities"])
    assert scheduler["current"]["rationale"]
    assert adaptive["state"] in ("OFFLINE_PRIOR", "LEARNING", "SHADOW",
                                "ADAPTIVE_ASSISTED", "ADAPTIVE_CONTROL")
    assert len(activity["grid"]) > 0
    client.post("/simulation/stop")


def test_metrics_and_models_endpoints(client):
    client.post("/simulation/start", json={"seed": 3, "speed": 0, "max_slots": 50})
    time.sleep(1.0)
    client.post("/simulation/stop")

    metrics = client.get("/metrics").json()
    assert "probability_of_detection" in metrics
    models = client.get("/models").json()
    assert "loaded" in models and "registry" in models


def test_pause_and_resume_endpoints(client):
    client.post("/simulation/start", json={"seed": 9, "speed": 20})
    time.sleep(0.4)
    assert client.post("/simulation/pause").json()["state"] == "PAUSED"
    assert client.post("/simulation/resume").json()["state"] == "RUNNING"
    client.post("/simulation/stop")


def test_experiment_job_runs_and_reports(client):
    job = client.post("/experiments/run", json={
        "scenario": "mixed", "seeds": [21], "strategies": ["random", "smart_scan"],
        "max_slots": 60,
    }).json()
    job_id = job["job_id"]

    deadline = time.time() + 90
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("FINISHED", "ERROR"):
            break
        time.sleep(0.3)
    assert job["status"] == "FINISHED", job.get("error")
    assert set(job["result"]["aggregate"]) == {"random", "smart_scan"}


# ------------------------------------------------------------------ adapters


def test_simulator_adapter_round_trip(small_env):
    receiver = ReceiverSimulator(small_env, seed=2)
    with SimulatorReceiverAdapter(receiver, run_id="r1") as adapter:
        assert adapter.get_capabilities()["n_bands"] == small_env.n_bands
        observation = adapter.execute(ScanCommand("r1", 0, 0))
        assert observation.outcome in set(Outcome)
        assert observation.source_id == "simulator"
        assert adapter.health_check()["healthy"]


def test_recorded_adapter_replays_a_stored_log(tmp_path, small_env):
    path = tmp_path / "replay.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for slot in range(5):
            fh.write(json.dumps({"band_id": slot % 3, "outcome": "HIT" if slot % 2 else "MISS",
                                 "quality": 0.8}) + "\n")

    with RecordedDataAdapter(path, run_id="replay") as adapter:
        assert adapter.get_capabilities()["records"] == 5
        outcomes = [adapter.execute(ScanCommand("replay", s, 0)).outcome for s in range(5)]
    assert outcomes[0] is Outcome.MISS and outcomes[1] is Outcome.HIT


def test_adapter_registry_lists_the_integration_options():
    assert set(ADAPTER_REGISTRY) == {"simulator", "recorded", "external"}


class _StubReceiverService(threading.Thread):
    """Minimal external receiver speaking the documented line-delimited JSON protocol."""

    def __init__(self):
        super().__init__(daemon=True)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self.commands = []

    def run(self) -> None:
        connection, _ = self._server.accept()
        with connection, connection.makefile("rwb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                request = json.loads(line)
                self.commands.append(request)
                command = request.get("cmd")
                if command == "capabilities":
                    reply = {"n_bands": 16, "capacity": 1, "scan_duration": 1}
                elif command == "scan":
                    # Odd bands report energy; a bare boolean, as the protocol allows.
                    reply = {"detected": bool(request["band_id"] % 2), "quality": 0.77}
                elif command == "status":
                    reply = {"state": "READY"}
                else:
                    reply = {"ok": True}
                stream.write((json.dumps(reply) + "\n").encode())
                stream.flush()


def test_external_adapter_speaks_the_documented_protocol():
    service = _StubReceiverService()
    service.start()

    adapter = ExternalReceiverAdapter(host="127.0.0.1", port=service.port, run_id="ext")
    with adapter:
        assert adapter.get_capabilities()["n_bands"] == 16
        assert adapter.health_check()["healthy"]

        detected = adapter.execute(ScanCommand("ext", 4, band_id=3))
        quiet = adapter.execute(ScanCommand("ext", 5, band_id=2))

    # A live source has no ground truth, so a detection maps to HIT and a
    # non-detection to CORRECT_NON_DETECTION.
    assert detected.outcome is Outcome.HIT
    assert detected.quality == pytest.approx(0.77)
    assert detected.source_id == "external"
    assert quiet.outcome is Outcome.CORRECT_NON_DETECTION
    assert [c["cmd"] for c in service.commands[:2]] == ["capabilities", "status"]


def test_engine_runs_through_an_external_adapter(cfg):
    """The intelligence core must not care where observations come from."""
    service = _StubReceiverService()
    service.start()

    adapter = ExternalReceiverAdapter(host="127.0.0.1", port=service.port, run_id="ext-run")
    engine = SmartScanEngine(cfg, strategy="smart_scan", seed=5, scenario="mixed",
                             persist=False, adapter=adapter)
    summary = engine.run(max_steps=40)

    assert summary["scans"] == 40
    assert any(c["cmd"] == "scan" for c in service.commands)
