"""Database engine, session factory and the batched run recorder.

Writes are buffered and flushed in batches, so a fast simulation loop is not
paced by disk I/O.  The recorder is the only object the engine talks to; it owns
the sampling policy that keeps full-grid state affordable.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import Config, database_url
from ..core.types import Decision, NormalizedObservation, band_label
from .models import (
    AdaptiveEventRecord,
    Base,
    Belief,
    BandRecord,
    DecisionRecord,
    Experiment,
    LogRecord,
    MetricRecord,
    ModelRecord,
    Observation,
    Prediction,
    Run,
    Scenario,
)

logger = logging.getLogger(__name__)

_engines: Dict[str, Engine] = {}
_factories: Dict[str, sessionmaker] = {}


def get_engine(cfg: Config, url: Optional[str] = None) -> Engine:
    """Process-wide engine per URL. SQLite needs check_same_thread disabled
    because the API, the simulation worker and the UI poll from different threads."""
    resolved = url or database_url(cfg)
    if resolved not in _engines:
        connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
        engine = create_engine(
            resolved,
            echo=bool(cfg.get_path("database.echo", False)),
            future=True,
            connect_args=connect_args,
        )
        Base.metadata.create_all(engine)
        _engines[resolved] = engine
        _factories[resolved] = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return _engines[resolved]


def get_session_factory(cfg: Config, url: Optional[str] = None) -> sessionmaker:
    get_engine(cfg, url)
    return _factories[url or database_url(cfg)]


@contextmanager
def session_scope(cfg: Config, url: Optional[str] = None) -> Iterator[Session]:
    factory = get_session_factory(cfg, url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(cfg: Config) -> Engine:
    """Create the schema if it does not exist yet."""
    return get_engine(cfg)


class RunRecorder:
    """Buffers one run's rows and flushes them in batches."""

    def __init__(self, cfg: Config, run_id: str, snapshot_stride: int = 25,
                 enabled: bool = True):
        self.cfg = cfg
        self.run_id = run_id
        self.enabled = enabled
        self.batch_size = int(cfg.get_path("database.batch_size", 200))
        self.snapshot_stride = max(1, int(snapshot_stride))
        self._buffer: List[Any] = []
        self._factory = get_session_factory(cfg) if enabled else None

    # ------------------------------------------------------------------ setup
    def start_run(self, *, strategy: str, seed: int, environment, config: Dict[str, Any],
                  model_id: Optional[str], model_kind: Optional[str],
                  adaptive_enabled: bool, source_id: str,
                  experiment_id: Optional[int] = None) -> None:
        if not self.enabled:
            return
        with session_scope(self.cfg) as session:
            scenario = Scenario(
                key=environment.scenario,
                title=environment.scenario,
                scenario_type=environment.scenario,
                n_bands=environment.n_bands,
                n_slots=environment.n_slots,
                seed=environment.seed,
                change_points=list(environment.change_points),
                params={"activity_rate": environment.activity_rate()},
            )
            session.add(scenario)
            session.flush()

            session.add(Run(
                id=self.run_id,
                scenario_id=scenario.id,
                experiment_id=experiment_id,
                strategy=strategy,
                seed=seed,
                n_bands=environment.n_bands,
                n_slots=environment.n_slots,
                model_id=model_id,
                model_kind=model_kind,
                adaptive_enabled=adaptive_enabled,
                source_id=source_id,
                status="RUNNING",
                config=config,
            ))
            rates = environment.truth.mean(axis=0)
            session.add_all([
                BandRecord(run_id=self.run_id, band_index=b, label=band_label(b),
                           truth_activity_rate=float(rates[b]))
                for b in range(environment.n_bands)
            ])

    # ------------------------------------------------------------------ record
    def record_step(self, decision: Decision, observation: NormalizedObservation,
                    reward: float, truth_active: Optional[bool],
                    ml_probabilities: Sequence[float], learner,
                    model_id: Optional[str], model_kind: str) -> None:
        if not self.enabled:
            return
        slot = decision.slot
        band = decision.band_id

        self._buffer.append(Observation(
            run_id=self.run_id, slot=slot, band_index=band,
            outcome=observation.outcome.value, detected=observation.detected,
            truth_active=truth_active, quality=observation.quality,
            scan_duration=observation.scan_duration,
            receiver_state=observation.receiver_state,
            source_id=observation.source_id, reward=float(reward),
        ))
        self._buffer.append(DecisionRecord(
            run_id=self.run_id, slot=slot, band_index=band, strategy=decision.strategy,
            sampled_value=decision.sampled_value, ml_probability=decision.ml_probability,
            posterior_mean=decision.posterior_mean, posterior_std=decision.posterior_std,
            exploration=decision.exploration, adaptive_influence=decision.adaptive_influence,
            candidates=[int(c) for c in decision.candidates], rationale=decision.rationale,
        ))
        self._buffer.append(Prediction(
            run_id=self.run_id, slot=slot, band_index=band,
            probability=float(ml_probabilities[band]), model_id=model_id,
            model_kind=model_kind, selected=True,
        ))
        self._buffer.append(Belief(
            run_id=self.run_id, slot=slot, band_index=band,
            alpha=decision.alpha, beta=decision.beta,
            posterior_mean=decision.posterior_mean, posterior_std=decision.posterior_std,
            selected=True,
        ))

        # Periodic full-grid snapshot keeps the run reconstructable without
        # writing every band at every slot.
        if slot % self.snapshot_stride == 0 and learner is not None:
            means, stds = learner.mean, learner.std
            alphas, betas = learner.alpha, learner.beta
            for b in range(len(ml_probabilities)):
                if b == band:
                    continue
                self._buffer.append(Prediction(
                    run_id=self.run_id, slot=slot, band_index=b,
                    probability=float(ml_probabilities[b]), model_id=model_id,
                    model_kind=model_kind, selected=False,
                ))
                self._buffer.append(Belief(
                    run_id=self.run_id, slot=slot, band_index=b,
                    alpha=float(alphas[b]), beta=float(betas[b]),
                    posterior_mean=float(means[b]), posterior_std=float(stds[b]),
                    selected=False,
                ))

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def record_adaptive_events(self, events: Sequence[Any]) -> None:
        if not self.enabled:
            return
        for event in events:
            self._buffer.append(AdaptiveEventRecord(
                run_id=self.run_id, slot=event.slot, event_type=event.event_type,
                detail=event.detail, payload=event.payload,
            ))

    def record_metrics(self, metrics: Dict[str, Any], slot: Optional[int] = None,
                       scope: str = "run") -> None:
        if not self.enabled:
            return
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._buffer.append(MetricRecord(
                    run_id=self.run_id, slot=slot, scope=scope,
                    name=name, value=float(value),
                ))

    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._buffer.append(LogRecord(run_id=self.run_id, level=level,
                                      message=message, context=context or {}))

    # ------------------------------------------------------------------ flush
    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return
        rows, self._buffer = self._buffer, []
        try:
            with session_scope(self.cfg) as session:
                session.add_all(rows)
        except Exception as exc:  # pragma: no cover - persistence must not kill a run
            logger.error("failed to flush %d rows for run %s: %s", len(rows), self.run_id, exc)

    def finish_run(self, status: str, summary: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.record_metrics(summary, scope="run")
        self.flush()
        try:
            with session_scope(self.cfg) as session:
                run = session.get(Run, self.run_id)
                if run is not None:
                    run.status = status
                    run.summary = _json_safe(summary)
                    run.finished_at = datetime.now(timezone.utc)
        except Exception as exc:  # pragma: no cover
            logger.error("failed to finalise run %s: %s", self.run_id, exc)


# --------------------------------------------------------------------- queries
def sync_model_registry(cfg: Config, registry) -> int:
    """Mirror registry entries into the database so the UI can list models."""
    rows = registry.list_models()
    with session_scope(cfg) as session:
        existing = {r for (r,) in session.execute(select(ModelRecord.model_id))}
        added = 0
        for row in rows:
            if row["model_id"] in existing:
                continue
            try:
                _, meta = registry.load(row["model_id"])
            except Exception:  # pragma: no cover - unreadable artifact
                continue
            session.add(ModelRecord(
                model_id=meta.model_id, kind=meta.kind, version=meta.version,
                feature_version=meta.feature_version, dataset_version=meta.dataset_version,
                trained_at=meta.trained_at, metrics=meta.validation_metrics,
                calibration=meta.calibration_metrics, hyperparameters=meta.hyperparameters,
            ))
            added += 1
    return added


def recent_runs(cfg: Config, limit: int = 25) -> List[Dict[str, Any]]:
    with session_scope(cfg) as session:
        rows = session.execute(
            select(Run).order_by(Run.started_at.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "run_id": r.id,
                "strategy": r.strategy,
                "seed": r.seed,
                "status": r.status,
                "n_slots": r.n_slots,
                "model_id": r.model_id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "summary": r.summary or {},
            }
            for r in rows
        ]


def run_detail(cfg: Config, run_id: str) -> Optional[Dict[str, Any]]:
    with session_scope(cfg) as session:
        run = session.get(Run, run_id)
        if run is None:
            return None
        events = session.execute(
            select(AdaptiveEventRecord).where(AdaptiveEventRecord.run_id == run_id)
            .order_by(AdaptiveEventRecord.slot)
        ).scalars().all()
        return {
            "run_id": run.id,
            "strategy": run.strategy,
            "seed": run.seed,
            "status": run.status,
            "summary": run.summary or {},
            "config": run.config or {},
            "adaptive_events": [
                {"slot": e.slot, "event_type": e.event_type, "detail": e.detail}
                for e in events
            ],
        }


def list_experiments(cfg: Config, limit: int = 25) -> List[Dict[str, Any]]:
    with session_scope(cfg) as session:
        rows = session.execute(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": e.id,
                "name": e.name,
                "scenario_key": e.scenario_key,
                "status": e.status,
                "seeds": e.seeds,
                "strategies": e.strategies,
                "summary": e.summary or {},
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in rows
        ]


def _json_safe(value: Any) -> Any:
    """Drop values SQLite's JSON encoder cannot represent (NaN, numpy scalars)."""
    import math

    import numpy as np

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


__all__ = [
    "get_engine", "get_session_factory", "session_scope", "init_database",
    "RunRecorder", "sync_model_registry", "recent_runs", "run_detail",
    "list_experiments",
]
