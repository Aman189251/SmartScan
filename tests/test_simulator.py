"""Environment and receiver simulator behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.core.types import Outcome, ScanCommand
from backend.app.simulator.environment import SCENARIOS, SyntheticEnvironment
from backend.app.simulator.receiver import ReceiverSimulator


def build_env(**kwargs) -> SyntheticEnvironment:
    params = dict(n_bands=12, n_slots=400, scenario="mixed", seed=7)
    params.update(kwargs)
    return SyntheticEnvironment(**params)


def test_same_seed_reproduces_the_environment():
    a, b = build_env(), build_env()
    assert np.array_equal(a.truth, b.truth)


def test_different_seed_changes_the_environment():
    assert not np.array_equal(build_env(seed=1).truth, build_env(seed=2).truth)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_produces_activity(scenario):
    env = build_env(scenario=scenario)
    assert env.truth.shape == (400, 12)
    assert 0.0 < env.activity_rate() < 0.9, "activity must be sparse but present"


def test_bursts_match_the_activity_grid():
    env = build_env(scenario="periodic")
    total_active = int(env.truth.sum())
    covered = sum(end - start for _, start, end in env.bursts())
    assert covered == total_active


def test_change_point_alters_the_active_band_set():
    env = build_env(scenario="shift", n_slots=800, change_points=[400])
    before = env.truth[:400].mean(axis=0)
    after = env.truth[400:].mean(axis=0)
    assert not np.allclose(before, after), "an environment shift must change the pattern"


def test_agile_activity_moves_between_bands():
    env = build_env(scenario="agile", n_slots=600)
    source = env.segments[0].sources[0]
    assert len(source.bands) > 1
    active_bands = {b for b in range(env.n_bands) if env.truth[:, b].any()}
    assert len(active_bands & set(source.bands)) > 1


# ---------------------------------------------------------------- receiver


def test_receiver_detection_statistics_match_configuration():
    env = build_env(n_slots=4000, scenario="periodic")
    receiver = ReceiverSimulator(env, pd=0.9, pfa=0.05, seed=3)
    hits = misses = false_alarms = quiet = 0
    for slot in range(4000):
        band = slot % env.n_bands
        outcome = receiver.observe(ScanCommand("t", slot, band)).outcome
        hits += outcome is Outcome.HIT
        misses += outcome is Outcome.MISS
        false_alarms += outcome is Outcome.FALSE_ALARM
        quiet += outcome is Outcome.CORRECT_NON_DETECTION

    measured_pd = hits / max(1, hits + misses)
    measured_pfa = false_alarms / max(1, false_alarms + quiet)
    assert measured_pd == pytest.approx(0.9, abs=0.05)
    assert measured_pfa == pytest.approx(0.05, abs=0.03)


def test_receiver_rejects_more_looks_than_its_capacity():
    env = build_env()
    receiver = ReceiverSimulator(env, capacity=1, seed=3)
    receiver.observe(ScanCommand("t", 0, 0))
    with pytest.raises(RuntimeError):
        receiver.observe(ScanCommand("t", 0, 1))


def test_capacity_two_allows_two_looks_in_one_slot():
    env = build_env()
    receiver = ReceiverSimulator(env, capacity=2, seed=3)
    receiver.observe(ScanCommand("t", 0, 0))
    receiver.observe(ScanCommand("t", 0, 1))
    with pytest.raises(RuntimeError):
        receiver.observe(ScanCommand("t", 0, 2))


def test_revisit_lockout_removes_a_band_from_availability():
    env = build_env()
    receiver = ReceiverSimulator(env, revisit_lockout=5, seed=3)
    receiver.observe(ScanCommand("t", 10, 3))
    assert 3 not in receiver.available_bands(12)
    assert 3 in receiver.available_bands(20)


def test_receiver_is_reproducible_for_a_seed():
    env = build_env()
    outcomes = []
    for _ in range(2):
        receiver = ReceiverSimulator(env, seed=11)
        outcomes.append([receiver.observe(ScanCommand("t", s, s % 12)).outcome
                         for s in range(200)])
    assert outcomes[0] == outcomes[1]
