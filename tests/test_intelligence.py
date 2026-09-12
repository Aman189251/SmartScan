"""Feature engine, Bayesian belief, adaptive layer and scheduler behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.adaptive.bayesian import BetaBernoulliLearner
from backend.app.adaptive.drift import (
    AdaptiveLearner,
    PageHinkley,
    ScoreMonitor,
    WindowedDriftDetector,
)
from backend.app.core.types import AdaptiveState
from backend.app.ml.features import FEATURE_NAMES, N_FEATURES, FeatureEngine
from backend.app.scheduler import BENCHMARK_ORDER, DecisionContext, build_scheduler

# ------------------------------------------------------------------ features


def test_feature_matrix_shape_and_names():
    engine = FeatureEngine(n_bands=8)
    matrix = engine.build_matrix(0)
    assert matrix.shape == (8, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES


def test_unobserved_bands_are_flagged_and_features_react_to_observations():
    engine = FeatureEngine(n_bands=4)
    never_index = FEATURE_NAMES.index("never_scanned")
    assert engine.build_matrix(0)[2, never_index] == 1.0

    engine.observe(0, 2, detected=True)
    row = engine.build_matrix(1)[2]
    assert row[never_index] == 0.0
    assert row[FEATURE_NAMES.index("ewma_fast")] > 0.5
    assert row[FEATURE_NAMES.index("last_detected")] == 1.0


def test_window_rates_track_recent_outcomes():
    engine = FeatureEngine(n_bands=2, short_window=4, medium_window=8, long_window=16)
    for slot in range(8):
        engine.observe(slot, 0, detected=slot >= 4)
    row = engine.build_matrix(9)[0]
    assert row[FEATURE_NAMES.index("rate_short")] == pytest.approx(1.0)
    assert row[FEATURE_NAMES.index("rate_long")] == pytest.approx(0.5)
    assert row[FEATURE_NAMES.index("trend_short_long")] > 0


def test_feature_engine_is_deterministic():
    def build() -> np.ndarray:
        engine = FeatureEngine(n_bands=6)
        for slot in range(50):
            engine.observe(slot, slot % 6, detected=slot % 3 == 0)
        return engine.build_matrix(50)

    assert np.array_equal(build(), build())


# ------------------------------------------------------------------ Bayesian


def test_prior_is_anchored_on_the_model_probability():
    learner = BetaBernoulliLearner(n_bands=3, prior_strength_k=10.0)
    learner.set_prior(np.array([0.8, 0.5, 0.2]))
    assert learner.alpha[0] == pytest.approx(8.0)
    assert learner.beta[0] == pytest.approx(2.0)
    assert learner.mean[0] == pytest.approx(0.8)


def test_hits_and_misses_move_the_posterior_in_the_right_direction():
    learner = BetaBernoulliLearner(n_bands=2, prior_strength_k=4.0)
    learner.set_prior(np.array([0.5, 0.5]))
    before = learner.mean[0]
    for _ in range(10):
        learner.update(0, True)
    assert learner.mean[0] > before
    for _ in range(20):
        learner.update(1, False)
    assert learner.mean[1] < before


def test_uncertainty_shrinks_as_evidence_accumulates():
    learner = BetaBernoulliLearner(n_bands=1, prior_strength_k=2.0)
    learner.set_prior(np.array([0.5]))
    wide = learner.std[0]
    for i in range(40):
        learner.update(0, i % 2 == 0)
    assert learner.std[0] < wide


def test_decay_forgets_old_evidence():
    learner = BetaBernoulliLearner(n_bands=1, prior_strength_k=1.0, decay_lambda=0.5)
    for _ in range(5):
        learner.update(0, True)
    strong = learner.hits[0]
    for _ in range(5):
        learner.decay()
    assert learner.hits[0] < strong / 8


def test_static_prior_mode_only_anchors_once():
    learner = BetaBernoulliLearner(n_bands=1, prior_strength_k=10.0, prior_mode="static")
    learner.set_prior(np.array([0.9]))
    learner.set_prior(np.array([0.1]))
    assert learner.alpha[0] == pytest.approx(9.0)


# ------------------------------------------------------------------- adaptive


def test_page_hinkley_flags_a_sustained_error_increase():
    detector = PageHinkley(delta=0.005, threshold=0.4, min_samples=20)
    rng = np.random.default_rng(0)
    for _ in range(60):
        assert not detector.update(float(rng.uniform(0.0, 0.05)))
    flagged = any(detector.update(0.9) for _ in range(40))
    assert flagged


def test_score_monitor_reports_quality():
    monitor = ScoreMonitor(window=50)
    for _ in range(25):
        monitor.add(0.9, True)
        monitor.add(0.1, False)
    assert monitor.brier < 0.02
    assert monitor.accuracy == pytest.approx(1.0)
    assert monitor.calibration_error < 0.15


def test_adaptive_layer_starts_in_shadow_and_never_influences_before_promotion(cfg):
    learner = AdaptiveLearner(n_bands=6, cfg=cfg)
    assert learner.state is AdaptiveState.OFFLINE_PRIOR
    assert learner.influence() == 0.0

    rng = np.random.default_rng(0)
    p_ml = np.full(6, 0.5)
    for slot in range(30):
        learner.shadow_probabilities(p_ml)
        learner.recommend(p_ml, rng)
        learner.observe(slot, slot % 6, detected=slot % 2 == 0,
                        p_offline=0.5, p_adaptive=0.5)
    assert learner.state in (AdaptiveState.LEARNING, AdaptiveState.SHADOW)
    assert learner.influence() == 0.0, "shadow mode must not influence the decision layer"


def test_windowed_detector_ignores_noise_but_catches_a_step_change():
    detector = WindowedDriftDetector(window=40, reference=120, z_threshold=3.0, min_gap=0.05)
    rng = np.random.default_rng(4)
    for _ in range(400):
        assert not detector.update(float(rng.normal(0.20, 0.05))), \
            "stationary noise must not raise an alarm"
    flagged = any(detector.update(float(rng.normal(0.55, 0.05))) for _ in range(200))
    assert flagged


def test_windowed_detector_ignores_a_drop_in_error():
    detector = WindowedDriftDetector(window=40, reference=120, z_threshold=3.0)
    rng = np.random.default_rng(4)
    for _ in range(200):
        detector.update(float(rng.normal(0.5, 0.05)))
    # Performance improving is not drift.
    assert not any(detector.update(float(rng.normal(0.05, 0.02))) for _ in range(200))


def test_drift_alarm_withdraws_influence_and_speeds_up_forgetting(cfg):
    cfg = cfg.merged({"adaptive": {
        "drift_signal": "model_error", "drift_window": 20,
        "drift_reference": 60, "drift_z_threshold": 2.5,
    }})
    learner = AdaptiveLearner(n_bands=4, cfg=cfg)
    learner.state = AdaptiveState.ADAPTIVE_ASSISTED
    baseline_decay = learner.shadow.decay_lambda

    events = []
    # A well-calibrated stretch: the model predicts activity and finds it.
    for slot in range(120):
        events += learner.observe(slot, slot % 4, detected=True,
                                  p_offline=0.95, p_adaptive=0.9)
    assert not any(e.event_type == "DRIFT_DETECTED" for e in events), \
        "a stable environment must not raise a drift alarm"

    # The environment changes underneath it: the same confident prediction now fails.
    for slot in range(120, 260):
        events += learner.observe(slot, slot % 4, detected=False,
                                  p_offline=0.95, p_adaptive=0.2)
    assert any(e.event_type == "DRIFT_DETECTED" for e in events)
    assert learner.state is AdaptiveState.SHADOW
    assert learner.shadow.decay_lambda < baseline_decay


# ------------------------------------------------------------------ schedulers


def make_context(n_bands: int = 8, available=None) -> DecisionContext:
    rng = np.random.default_rng(5)
    learner = BetaBernoulliLearner(n_bands=n_bands, prior_strength_k=6.0)
    p_ml = np.linspace(0.1, 0.9, n_bands)
    learner.set_prior(p_ml)
    return DecisionContext(
        slot=25,
        n_bands=n_bands,
        available_bands=list(range(n_bands)) if available is None else list(available),
        rng=rng,
        ml_probabilities=p_ml,
        scans=np.ones(n_bands) * 4,
        detections=np.arange(n_bands, dtype=float) / 2.0,
        learner=learner,
        adaptive=None,
        metadata={"since_scan": np.arange(n_bands, dtype=float)},
    )


@pytest.mark.parametrize("name", BENCHMARK_ORDER)
def test_every_scheduler_returns_a_valid_available_band(name, cfg):
    scheduler = build_scheduler(name, cfg)
    ctx = make_context(available=[1, 3, 5, 7])
    for _ in range(20):
        decision = scheduler.select(ctx)
        assert decision.band_id in ctx.available_bands
        assert decision.strategy == name
        assert 0.0 <= decision.posterior_mean <= 1.0


def test_greedy_ml_takes_the_highest_probability():
    scheduler = build_scheduler("greedy_ml")
    ctx = make_context()
    assert scheduler.select(ctx).band_id == ctx.n_bands - 1


def test_round_robin_sweeps_every_band_in_order():
    scheduler = build_scheduler("round_robin")
    ctx = make_context(n_bands=5)
    assert [scheduler.select(ctx).band_id for _ in range(7)] == [0, 1, 2, 3, 4, 0, 1]


def test_smart_scan_selects_from_its_candidate_set(cfg):
    scheduler = build_scheduler("smart_scan", cfg)
    ctx = make_context(n_bands=10)
    for _ in range(30):
        decision = scheduler.select(ctx)
        if not decision.exploration:
            assert decision.band_id in decision.candidates
            assert len(decision.candidates) <= int(cfg.get_path("ml.candidate_k", 5)) + 1


def test_smart_scan_explains_itself(cfg):
    decision = build_scheduler("smart_scan", cfg).select(make_context())
    assert decision.rationale
    assert decision.scores
