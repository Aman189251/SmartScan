"""Smart Scan engine: one object that runs the closed loop.

    state + history -> features -> offline probability -> candidate set
      -> Bayesian posterior -> Thompson selection -> receiver adapter
      -> normalised observation -> persistence -> adaptive learner -> repeat

Reproducibility: a run is fully determined by its seed plus its configuration.
Environment, receiver and decision randomness are drawn from separate seeded
generators derived from that one seed, so replaying a run reproduces every
decision.

Ground-truth barrier: ``environment.is_active`` is read in exactly one place in
this file, and the value goes only to the evaluation engine.  It is never put
into a decision context, a feature vector or an adaptive update.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..adaptive.bayesian import BetaBernoulliLearner
from ..adaptive.drift import AdaptiveLearner
from ..config import Config
from ..core.types import (
    AdaptiveState,
    Decision,
    NormalizedObservation,
    RunState,
    ScanCommand,
    StepResult,
    band_label,
)
from ..db.database import RunRecorder
from ..evaluation.metrics import EvaluationEngine
from ..ml.features import FeatureEngine
from ..ml.predict import BasePredictor, load_predictor
from ..receiver.adapter import ReceiverAdapter, SimulatorReceiverAdapter
from ..scheduler import DecisionContext, build_scheduler
from ..simulator.receiver import ReceiverSimulator
from ..simulator.scenarios import build_environment

logger = logging.getLogger(__name__)

_TIMELINE_LIMIT = 400


class SmartScanEngine:
    """Owns one run: environment, receiver, intelligence layers and records."""

    def __init__(
        self,
        cfg: Config,
        strategy: Optional[str] = None,
        seed: Optional[int] = None,
        scenario: Optional[str] = None,
        adaptive_enabled: Optional[bool] = None,
        persist: bool = True,
        predictor: Optional[BasePredictor] = None,
        adapter: Optional[ReceiverAdapter] = None,
        run_id: Optional[str] = None,
        experiment_id: Optional[int] = None,
    ):
        self.cfg = cfg
        self.strategy = strategy or str(cfg.get_path("decision.strategy", "smart_scan"))
        self.scenario_key = scenario or str(cfg.get_path("environment.scenario", "mixed"))
        self.seed = int(seed if seed is not None else cfg.get_path("environment.seed", 1234))
        self.adaptive_enabled = bool(
            cfg.get_path("adaptive.enabled", True) if adaptive_enabled is None else adaptive_enabled
        )
        self.persist = persist
        self.run_id = run_id or f"{self.strategy}-{uuid.uuid4().hex[:10]}"
        self.experiment_id = experiment_id
        self._external_predictor = predictor
        self._external_adapter = adapter

        self.state = RunState.IDLE
        self.error: Optional[str] = None
        self.slot = 0
        self.started_at: Optional[float] = None
        self._timeline: List[Dict[str, Any]] = []
        self._last_step: Optional[StepResult] = None

        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        cfg = self.cfg
        self.environment = build_environment(cfg, seed=self.seed, preset_key=self.scenario_key)
        self.n_bands = self.environment.n_bands
        self.n_slots = self.environment.n_slots
        self.horizon = int(cfg.get_path("ml.horizon", 1))
        self.capacity = max(1, int(cfg.get_path("receiver.capacity", 1)))

        self.receiver = ReceiverSimulator(
            self.environment,
            pd=float(cfg.get_path("receiver.pd", 0.90)),
            pfa=float(cfg.get_path("receiver.pfa", 0.03)),
            capacity=self.capacity,
            scan_duration=int(cfg.get_path("receiver.scan_duration", 1)),
            revisit_lockout=int(cfg.get_path("receiver.revisit_lockout", 0)),
            seed=self.seed + 991,
        )
        self.adapter: ReceiverAdapter = self._external_adapter or SimulatorReceiverAdapter(
            self.receiver, run_id=self.run_id
        )
        self.adapter.connect()
        self.adapter.configure_scan(run_id=self.run_id)

        self.features = FeatureEngine(
            n_bands=self.n_bands,
            short_window=int(cfg.get_path("features.short_window", 8)),
            medium_window=int(cfg.get_path("features.medium_window", 24)),
            long_window=int(cfg.get_path("features.long_window", 64)),
            ewma_fast=float(cfg.get_path("features.ewma_fast", 0.30)),
            ewma_slow=float(cfg.get_path("features.ewma_slow", 0.05)),
            cycle_length=int(cfg.get_path("features.cycle_length", 120)),
        )
        self.predictor = self._external_predictor or load_predictor("xgboost")
        self.learner = BetaBernoulliLearner(
            n_bands=self.n_bands,
            prior_strength_k=float(cfg.get_path("bayesian.prior_strength_k", 10.0)),
            decay_lambda=float(cfg.get_path("bayesian.decay_lambda", 0.995)),
            prior_mode=str(cfg.get_path("bayesian.prior_mode", "dynamic")),
            min_evidence=float(cfg.get_path("bayesian.min_evidence", 0.0)),
        )
        self.adaptive = AdaptiveLearner(self.n_bands, cfg, enabled=self.adaptive_enabled)
        self.scheduler = build_scheduler(self.strategy, cfg)
        self.scheduler.reset()
        self.evaluator = EvaluationEngine(self.environment, cfg)
        self.rng = np.random.default_rng(self.seed + 13)

        model_info = self.predictor.describe()
        self.recorder = RunRecorder(
            cfg, self.run_id,
            snapshot_stride=max(1, self.n_slots // 100),
            enabled=self.persist,
        )
        self.recorder.start_run(
            strategy=self.strategy,
            seed=self.seed,
            environment=self.environment,
            config={
                "scenario": self.scenario_key,
                "receiver": {"pd": self.receiver.pd, "pfa": self.receiver.pfa,
                             "capacity": self.capacity},
                "bayesian": dict(cfg.get_path("bayesian", {}) or {}),
                "decision": dict(cfg.get_path("decision", {}) or {}),
                "adaptive_enabled": self.adaptive_enabled,
                "model": model_info,
            },
            model_id=model_info.get("model_id"),
            model_kind=model_info.get("kind", "heuristic"),
            adaptive_enabled=self.adaptive_enabled,
            source_id=getattr(self.adapter, "source_id", "simulator"),
            experiment_id=self.experiment_id,
        )

    # ------------------------------------------------------------------- loop
    def step(self) -> Optional[StepResult]:
        """Execute one decision cycle. Returns None when the run is complete."""
        if self.slot >= self.n_slots - self.horizon:
            self.finish()
            return None
        if self.started_at is None:
            self.started_at = time.time()
            self.state = RunState.RUNNING

        result: Optional[StepResult] = None
        for _ in range(self.capacity):
            result = self._decision_cycle(self.slot)
            if result is None:
                break
        self.slot += 1
        return result

    def _decision_cycle(self, slot: int) -> Optional[StepResult]:
        available = self.receiver.available_bands(slot)
        if not available:
            return None

        # --- Layer 1: offline prediction --------------------------------
        feature_matrix = self.features.build_matrix(slot)
        p_ml = np.asarray(self.predictor.predict(feature_matrix), dtype=float)

        # The adaptive layer forms its own view every slot, whether or not it is
        # currently allowed to influence anything.
        p_shadow = (self.adaptive.shadow_probabilities(p_ml)
                    if self.adaptive.enabled else p_ml)
        if self.adaptive.enabled:
            self.adaptive.recommend(p_ml, self.rng)

        # --- Layer 2: mathematical decision ------------------------------
        since_scan = np.where(self.features.last_scan >= 0,
                              slot - self.features.last_scan, 10_000)
        ctx = DecisionContext(
            slot=slot,
            n_bands=self.n_bands,
            available_bands=list(available),
            rng=self.rng,
            ml_probabilities=p_ml,
            scans=self.features.scans,
            detections=self.features.detections,
            learner=self.learner,
            adaptive=self.adaptive if self.adaptive.enabled else None,
            features=feature_matrix,
            metadata={"since_scan": since_scan},
        )
        decision: Decision = self.scheduler.select(ctx)
        band = decision.band_id

        # --- Layer 3: scheduling and execution ---------------------------
        observation: NormalizedObservation = self.adapter.execute(
            ScanCommand(run_id=self.run_id, slot=slot, band_id=band,
                        dwell=int(self.receiver.constraints.scan_duration))
        )

        # --- Evaluation (the only ground-truth read in the live path) -----
        truth_active = bool(self.environment.is_active(slot, band))
        reward = self.evaluator.record(
            slot=slot, band_id=band, outcome=observation.outcome,
            ml_probability=float(p_ml[band]), posterior_mean=decision.posterior_mean,
            exploration=decision.exploration,
        )

        # --- Learning ------------------------------------------------------
        self.features.observe(slot, band, observation.detected)
        self.learner.update(band, observation.detected)
        self.learner.decay()
        events = self.adaptive.observe(
            slot=slot, band_id=band, detected=observation.detected,
            p_offline=float(p_ml[band]), p_adaptive=float(p_shadow[band]),
        )

        # --- Persistence ---------------------------------------------------
        model_info = self.predictor.describe()
        self.recorder.record_step(
            decision=decision, observation=observation, reward=reward,
            truth_active=truth_active, ml_probabilities=p_ml, learner=self.learner,
            model_id=model_info.get("model_id"), model_kind=model_info.get("kind", "heuristic"),
        )
        if events:
            self.recorder.record_adaptive_events(events)

        step = StepResult(
            slot=slot,
            decision=decision,
            observation=observation,
            reward=reward,
            truth_active=truth_active,
            ml_probabilities=p_ml.round(4).tolist(),
            posterior_means=self.learner.mean.round(4).tolist(),
            posterior_stds=self.learner.std.round(4).tolist(),
            adaptive_state=self.adaptive.state,
            drift_flag=self.adaptive.drift_flag,
            metrics={
                "rolling_detection_rate": round(self.evaluator.rolling_detection_rate, 4),
                "cumulative_reward": round(self.evaluator.cumulative_reward, 3),
                "scans": self.evaluator.n_scans,
            },
        )
        self._last_step = step
        self._timeline.append({
            "slot": slot,
            "band": band,
            "label": band_label(band),
            "outcome": observation.outcome.value,
            "detected": observation.detected,
            "ml_probability": round(float(p_ml[band]), 4),
            "posterior_mean": round(decision.posterior_mean, 4),
            "sampled": round(decision.sampled_value, 4),
            "exploration": decision.exploration,
        })
        if len(self._timeline) > _TIMELINE_LIMIT:
            del self._timeline[:-_TIMELINE_LIMIT]
        return step

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run to completion (or ``max_steps`` slots) and return the summary."""
        steps = 0
        limit = max_steps if max_steps is not None else self.n_slots
        while steps < limit and self.state not in (RunState.FINISHED, RunState.STOPPED,
                                                   RunState.ERROR):
            if self.step() is None:
                break
            steps += 1
        if self.state is not RunState.FINISHED and self.slot >= self.n_slots - self.horizon:
            self.finish()
        return self.summary()

    # ---------------------------------------------------------------- control
    def pause(self) -> None:
        if self.state is RunState.RUNNING:
            self.state = RunState.PAUSED

    def resume(self) -> None:
        if self.state is RunState.PAUSED:
            self.state = RunState.RUNNING

    def stop(self) -> None:
        if self.state in (RunState.FINISHED, RunState.STOPPED):
            return
        self.state = RunState.STOPPED
        self._close("STOPPED")

    def finish(self) -> None:
        if self.state is RunState.FINISHED:
            return
        self.state = RunState.FINISHED
        self._close("FINISHED")

    def _close(self, status: str) -> None:
        try:
            self.adapter.disconnect()
        except Exception:  # pragma: no cover - adapter teardown must not raise
            pass
        summary = self.evaluator.summary(self.environment.change_points)
        summary["adaptive_state"] = self.adaptive.state.value
        summary["drift_events"] = self.adaptive.drift_count
        self.recorder.finish_run(status, summary)

    # ---------------------------------------------------------------- reports
    def summary(self) -> Dict[str, Any]:
        summary = self.evaluator.summary(self.environment.change_points)
        summary.update({
            "run_id": self.run_id,
            "strategy": self.strategy,
            "scenario": self.scenario_key,
            "seed": self.seed,
            "state": self.state.value,
            "slot": self.slot,
            "n_slots": self.n_slots,
            "adaptive_state": self.adaptive.state.value,
            "drift_events": self.adaptive.drift_count,
            "model": self.predictor.describe(),
            "elapsed_seconds": round(time.time() - self.started_at, 2) if self.started_at else 0.0,
        })
        return summary

    def status(self) -> Dict[str, Any]:
        """Light-weight state for polling. No large arrays."""
        step = self._last_step
        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "strategy": self.strategy,
            "scenario": self.scenario_key,
            "seed": self.seed,
            "slot": self.slot,
            "n_slots": self.n_slots,
            "progress": round(self.slot / max(1, self.n_slots), 4),
            "scans": self.evaluator.n_scans,
            "rolling_detection_rate": round(self.evaluator.rolling_detection_rate, 4),
            "cumulative_reward": round(self.evaluator.cumulative_reward, 3),
            "adaptive_state": self.adaptive.state.value,
            "adaptive_influence": round(self.adaptive.influence(), 3),
            "drift_flag": self.adaptive.drift_flag,
            "drift_events": self.adaptive.drift_count,
            "model": self.predictor.describe(),
            "receiver": self.receiver.describe(),
            "current": {
                "slot": step.slot,
                "band": step.decision.band_id,
                "label": band_label(step.decision.band_id),
                "outcome": step.observation.outcome.value,
                "ml_probability": round(step.decision.ml_probability, 4),
                "posterior_mean": round(step.decision.posterior_mean, 4),
                "posterior_std": round(step.decision.posterior_std, 4),
                "sampled": round(step.decision.sampled_value, 4),
                "alpha": round(step.decision.alpha, 3),
                "beta": round(step.decision.beta, 3),
                "candidates": step.decision.candidates,
                "scores": step.decision.scores,
                "exploration": step.decision.exploration,
                "adaptive_influence": round(step.decision.adaptive_influence, 3),
                "rationale": step.decision.rationale,
            } if step is not None else None,
            "error": self.error,
        }

    def state_snapshot(self, include_grid: bool = True) -> Dict[str, Any]:
        """Full snapshot for the desktop UI."""
        snapshot: Dict[str, Any] = self.status()
        snapshot["timeline"] = list(self._timeline[-120:])
        snapshot["adaptive"] = self.adaptive.status()
        snapshot["bands"] = self.evaluator.band_statistics()
        snapshot["curves"] = self.evaluator.curves()
        snapshot["metrics"] = self.evaluator.summary(self.environment.change_points)
        if include_grid and self._last_step is not None:
            snapshot["ml_probabilities"] = self._last_step.ml_probabilities
            snapshot["posterior_means"] = self._last_step.posterior_means
            snapshot["posterior_stds"] = self._last_step.posterior_stds
        return snapshot

    def activity_window(self, width: int = 200) -> Dict[str, Any]:
        """Recent slice of the hidden activity grid, for display only.

        This is a visualisation of the simulated environment, exactly as the
        design document's spectrum/time view calls for.  It is produced for the
        operator after the fact and is never routed back into a decision.
        """
        end = max(1, self.slot)
        start = max(0, end - width)
        grid = self.environment.truth[start:end].T
        scans: List[List[int]] = []
        for entry in self._timeline:
            if start <= entry["slot"] < end:
                scans.append([entry["slot"] - start, entry["band"],
                              1 if entry["detected"] else 0])
        return {
            "start_slot": start,
            "end_slot": end,
            "n_bands": self.n_bands,
            "grid": grid.astype(int).tolist(),
            "scans": scans,
            "change_points": [cp - start for cp in self.environment.change_points
                              if start <= cp < end],
        }


__all__ = ["SmartScanEngine"]
