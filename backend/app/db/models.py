"""SQLAlchemy ORM models.

SQLite is the store: embedded, offline, no service to install, and a single file
an operator can copy off the workstation.  SQLAlchemy keeps the schema in one
place so nothing else in the codebase writes SQL.

Volume note: a run produces one decision per slot but a probability and a
posterior for every band at every slot.  Persisting the full grid would dominate
the database, so the recorder stores the acted-on band every slot and a complete
snapshot of all bands on a configurable stride.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(128), default="")
    scenario_type: Mapped[str] = mapped_column(String(32), default="mixed")
    n_bands: Mapped[int] = mapped_column(Integer, default=32)
    n_slots: Mapped[int] = mapped_column(Integer, default=2000)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    change_points: Mapped[Optional[dict]] = mapped_column(JSON, default=list)
    params: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="scenario")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    scenario_key: Mapped[str] = mapped_column(String(64), default="mixed")
    seeds: Mapped[Optional[dict]] = mapped_column(JSON, default=list)
    strategies: Mapped[Optional[dict]] = mapped_column(JSON, default=list)
    config: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    summary: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="RUNNING")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    runs: Mapped[list["Run"]] = relationship(back_populates="experiment")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    scenario_id: Mapped[Optional[int]] = mapped_column(ForeignKey("scenarios.id"), nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    n_bands: Mapped[int] = mapped_column(Integer, default=32)
    n_slots: Mapped[int] = mapped_column(Integer, default=0)
    model_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_kind: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    adaptive_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    source_id: Mapped[str] = mapped_column(String(32), default="simulator")
    status: Mapped[str] = mapped_column(String(16), default="RUNNING")
    config: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    summary: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    scenario: Mapped[Optional[Scenario]] = relationship(back_populates="runs")
    experiment: Mapped[Optional[Experiment]] = relationship(back_populates="runs")


class BandRecord(Base):
    __tablename__ = "bands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    band_index: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(16))
    truth_activity_rate: Mapped[float] = mapped_column(Float, default=0.0)


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    band_index: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String(24))
    detected: Mapped[bool] = mapped_column(Boolean, default=False)
    truth_active: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    quality: Mapped[float] = mapped_column(Float, default=1.0)
    scan_duration: Mapped[int] = mapped_column(Integer, default=1)
    receiver_state: Mapped[str] = mapped_column(String(24), default="READY")
    source_id: Mapped[str] = mapped_column(String(32), default="simulator")
    reward: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (Index("ix_observations_run_slot", "run_id", "slot"),)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    band_index: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    model_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_kind: Mapped[str] = mapped_column(String(32), default="heuristic")
    selected: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_predictions_run_slot", "run_id", "slot"),)


class Belief(Base):
    __tablename__ = "beliefs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    band_index: Mapped[int] = mapped_column(Integer)
    alpha: Mapped[float] = mapped_column(Float)
    beta: Mapped[float] = mapped_column(Float)
    posterior_mean: Mapped[float] = mapped_column(Float)
    posterior_std: Mapped[float] = mapped_column(Float)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_beliefs_run_slot", "run_id", "slot"),)


class DecisionRecord(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    band_index: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(32))
    sampled_value: Mapped[float] = mapped_column(Float, default=0.0)
    ml_probability: Mapped[float] = mapped_column(Float, default=0.0)
    posterior_mean: Mapped[float] = mapped_column(Float, default=0.0)
    posterior_std: Mapped[float] = mapped_column(Float, default=0.0)
    exploration: Mapped[bool] = mapped_column(Boolean, default=False)
    adaptive_influence: Mapped[float] = mapped_column(Float, default=0.0)
    candidates: Mapped[Optional[dict]] = mapped_column(JSON, default=list)
    rationale: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (Index("ix_decisions_run_slot", "run_id", "slot"),)


class AdaptiveEventRecord(Base):
    __tablename__ = "adaptive_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MetricRecord(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    slot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    scope: Mapped[str] = mapped_column(String(16), default="run")
    name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(16), default="1.0")
    feature_version: Mapped[str] = mapped_column(String(16), default="")
    dataset_version: Mapped[str] = mapped_column(String(16), default="")
    trained_at: Mapped[str] = mapped_column(String(32), default="")
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    calibration: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)


class LogRecord(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(48), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    level: Mapped[str] = mapped_column(String(16), default="INFO")
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)


ALL_TABLES = [
    Scenario, Experiment, Run, BandRecord, Observation, Prediction, Belief,
    DecisionRecord, AdaptiveEventRecord, MetricRecord, ModelRecord, LogRecord,
]

__all__ = [
    "Base", "Scenario", "Experiment", "Run", "BandRecord", "Observation",
    "Prediction", "Belief", "DecisionRecord", "AdaptiveEventRecord",
    "MetricRecord", "ModelRecord", "LogRecord", "ALL_TABLES",
]
