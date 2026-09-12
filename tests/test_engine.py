"""Engine, evaluation and persistence: the closed loop end to end."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.core.engine import SmartScanEngine
from backend.app.core.types import Outcome, RunState
from backend.app.db.database import recent_runs, session_scope
from backend.app.db.models import DecisionRecord, Observation
from backend.app.evaluation.benchmark import run_benchmark
from backend.app.evaluation.metrics import EvaluationEngine
from backend.app.ml.predict import HeuristicPredictor
from backend.app.scheduler import BENCHMARK_ORDER


def build_engine(cfg, **kwargs) -> SmartScanEngine:
    params = dict(strategy="smart_scan", seed=17, scenario="mixed", persist=False,
                  predictor=HeuristicPredictor())
    params.update(kwargs)
    return SmartScanEngine(cfg, **params)


def test_engine_completes_a_run_and_reports_metrics(cfg):
    summary = build_engine(cfg).run()
    assert summary["scans"] > 0
    assert summary["state"] == RunState.FINISHED.value
    for key in ("probability_of_detection", "probability_of_false_alarm",
                "average_intercept_rate", "intercept_ratio", "coverage",
                "cumulative_reward", "percentage_correct_predictions"):
        assert key in summary


def test_runs_are_reproducible_from_the_seed(cfg):
    first = build_engine(cfg).run()
    second = build_engine(cfg).run()
    for key in ("hits", "misses", "false_alarms", "cumulative_reward",
                "average_intercept_rate"):
        assert first[key] == second[key], f"{key} differed between identical seeds"


def test_a_different_seed_produces_a_different_run(cfg):
    assert build_engine(cfg, seed=1).run()["hits"] != build_engine(cfg, seed=2).run()["hits"] \
        or build_engine(cfg, seed=1).run()["coverage"] != build_engine(cfg, seed=2).run()["coverage"]


@pytest.mark.parametrize("strategy", BENCHMARK_ORDER)
def test_every_strategy_runs_end_to_end(cfg, strategy):
    summary = build_engine(cfg, strategy=strategy).run()
    assert summary["scans"] > 0
    assert 0.0 <= summary["coverage"] <= 1.0


def test_scheduler_never_receives_ground_truth(cfg):
    """The decision context must expose observations only."""
    engine = build_engine(cfg)
    captured = {}
    original = engine.scheduler.select

    def spy(ctx):
        captured["ctx"] = ctx
        return original(ctx)

    engine.scheduler.select = spy
    engine.step()

    ctx = captured["ctx"]
    assert not hasattr(ctx, "environment")
    assert "truth" not in ctx.metadata
    for value in ctx.metadata.values():
        assert not isinstance(value, np.ndarray) or value.shape != engine.environment.truth.shape
    assert ctx.learner is engine.learner


def test_feature_engine_only_sees_scanned_bands(cfg):
    engine = build_engine(cfg)
    for _ in range(25):
        engine.step()
    scanned = engine.features.scans
    assert scanned.sum() == 25
    assert (scanned > 0).sum() <= 25, "only executed scans may enter the feature history"


def test_measured_receiver_statistics_match_the_configuration(cfg):
    summary = build_engine(cfg, strategy="round_robin").run()
    assert summary["probability_of_detection"] == pytest.approx(
        float(cfg.get_path("receiver.pd")), abs=0.12)
    assert summary["probability_of_false_alarm"] == pytest.approx(
        float(cfg.get_path("receiver.pfa")), abs=0.06)


def test_pause_and_stop_control_the_run_state(cfg):
    engine = build_engine(cfg)
    engine.step()
    engine.pause()
    assert engine.state is RunState.PAUSED
    engine.resume()
    assert engine.state is RunState.RUNNING
    engine.stop()
    assert engine.state is RunState.STOPPED


def test_persistence_writes_observations_and_decisions(cfg):
    engine = build_engine(cfg, persist=True)
    engine.run(max_steps=60)
    with session_scope(cfg) as session:
        observations = session.query(Observation).filter_by(run_id=engine.run_id).count()
        decisions = session.query(DecisionRecord).filter_by(run_id=engine.run_id).count()
    assert observations == decisions > 0
    runs = recent_runs(cfg, limit=5)
    assert any(r["run_id"] == engine.run_id for r in runs)


def test_activity_window_is_display_only(cfg):
    engine = build_engine(cfg)
    engine.run(max_steps=40)
    window = engine.activity_window(width=30)
    assert window["end_slot"] - window["start_slot"] <= 30
    assert len(window["grid"]) == engine.n_bands


# ------------------------------------------------------------------ evaluation


def test_evaluation_counts_outcomes_and_delays(small_env, cfg):
    evaluator = EvaluationEngine(small_env, cfg)
    # Pick a burst long enough that a look two slots in still falls inside it.
    band, start, _ = next(b for b in small_env.bursts() if b[2] - b[1] >= 4)
    evaluator.record(start + 2, band, Outcome.HIT, 0.7, 0.6)
    evaluator.record(start + 3, band, Outcome.MISS, 0.7, 0.6)
    summary = evaluator.summary()
    assert summary["hits"] == 1
    assert summary["bursts_intercepted"] == 1
    assert summary["average_intercept_delay"] == pytest.approx(2.0)
    assert summary["probability_of_detection"] == pytest.approx(0.5)


def test_reward_follows_the_configured_weights(small_env, cfg):
    evaluator = EvaluationEngine(small_env, cfg)
    weights = evaluator.weights
    assert evaluator.weights.reward(Outcome.HIT) == pytest.approx(
        weights.detection_reward - weights.scan_cost)
    assert evaluator.weights.reward(Outcome.FALSE_ALARM) == pytest.approx(
        -weights.false_alarm_penalty - weights.scan_cost)


# ------------------------------------------------------------------ benchmark


def test_benchmark_compares_policies_on_identical_environments(cfg):
    result = run_benchmark(cfg, scenario="mixed", seeds=[11, 12],
                           strategies=["random", "round_robin", "smart_scan"],
                           persist=False, max_slots=150)
    assert set(result["aggregate"]) == {"random", "round_robin", "smart_scan"}
    for stats in result["aggregate"].values():
        assert stats["runs"] == 2
        assert "average_intercept_rate_mean" in stats
    assert len(result["ranking"]) == 3
