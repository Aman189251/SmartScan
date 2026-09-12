"""Evaluation engine.

This is the only operational component allowed to read hidden ground truth, and
it reads it strictly after the fact.  It never returns anything to the scheduler
during a run.

The metric set is organised around the figures of merit named in the problem
statement, with the exact definition used stated for each one:

probability_of_detection      HIT / (HIT + MISS)                    - measured Pd
probability_of_false_alarm    FALSE_ALARM / (FA + correct rejection) - measured Pfa
sensitivity_index             Pd - Pfa (Youden's J)
average_intercept_rate        detections per time slot
intercept_ratio               activity bursts intercepted at least once
average_intercept_delay       mean slots from activity onset to first interception
intercept_time_error_rms      root-mean-square of those delays
percentage_correct_predictions accuracy of the model probability at 0.5, scored
                              against the realised observation
average_reward                mean configured reward per slot
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Outcome


@dataclass(slots=True)
class StepRecord:
    slot: int
    band_id: int
    outcome: Outcome
    reward: float
    ml_probability: float
    posterior_mean: float
    truth_active: bool
    exploration: bool = False


@dataclass(slots=True)
class RewardWeights:
    detection_reward: float = 1.0
    false_alarm_penalty: float = 0.5
    miss_penalty: float = 0.2
    scan_cost: float = 0.01

    @classmethod
    def from_config(cls, cfg) -> "RewardWeights":
        block = cfg.get_path("reward", {}) or {}
        get = block.get if hasattr(block, "get") else dict(block).get
        return cls(
            detection_reward=float(get("detection_reward", 1.0)),
            false_alarm_penalty=float(get("false_alarm_penalty", 0.5)),
            miss_penalty=float(get("miss_penalty", 0.2)),
            scan_cost=float(get("scan_cost", 0.01)),
        )

    def reward(self, outcome: Outcome) -> float:
        value = -self.scan_cost
        if outcome is Outcome.HIT:
            value += self.detection_reward
        elif outcome is Outcome.FALSE_ALARM:
            value -= self.false_alarm_penalty
        elif outcome is Outcome.MISS:
            value -= self.miss_penalty
        return value


class EvaluationEngine:
    """Accumulates run records and computes the full metric set."""

    def __init__(self, environment, cfg, rolling_window: int = 100):
        self.environment = environment
        self.n_bands = environment.n_bands
        self.weights = RewardWeights.from_config(cfg)
        self.rolling_window = int(cfg.get_path("evaluation.rolling_window", rolling_window))
        self.adaptation_tolerance = float(cfg.get_path("evaluation.adaptation_tolerance", 0.9))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.records: List[StepRecord] = []
        self.counts = {o: 0 for o in Outcome}
        self.band_scans = np.zeros(self.n_bands, dtype=np.int64)
        self.band_hits = np.zeros(self.n_bands, dtype=np.int64)
        self.cumulative_reward = 0.0
        self._rolling: Deque[int] = deque(maxlen=self.rolling_window)
        self._reward_curve: List[float] = []
        self._rolling_curve: List[float] = []
        # Bursts grouped per band and kept sorted, so marking an interception is
        # a binary search rather than a scan over every burst in the run.
        self._bursts_by_band: Dict[int, List[tuple]] = {}
        self._burst_count = 0
        for band, start, end in self.environment.bursts():
            self._bursts_by_band.setdefault(band, []).append((start, end))
            self._burst_count += 1
        for entries in self._bursts_by_band.values():
            entries.sort()
        self._burst_starts = {b: [s for s, _ in v] for b, v in self._bursts_by_band.items()}
        self._intercepted: Dict[tuple, int] = {}

    # ----------------------------------------------------------------- record
    def record(self, slot: int, band_id: int, outcome: Outcome, ml_probability: float,
               posterior_mean: float, exploration: bool = False) -> float:
        truth = bool(self.environment.is_active(slot, band_id))
        reward = self.weights.reward(outcome)
        self.records.append(StepRecord(
            slot=slot, band_id=band_id, outcome=outcome, reward=reward,
            ml_probability=float(ml_probability), posterior_mean=float(posterior_mean),
            truth_active=truth, exploration=exploration,
        ))
        self.counts[outcome] += 1
        self.band_scans[band_id] += 1
        self.band_hits[band_id] += int(outcome is Outcome.HIT)
        self.cumulative_reward += reward
        self._rolling.append(int(outcome is Outcome.HIT))
        self._reward_curve.append(self.cumulative_reward)
        self._rolling_curve.append(self.rolling_detection_rate)

        if outcome is Outcome.HIT:
            self._mark_intercept(slot, band_id)
        return reward

    def _mark_intercept(self, slot: int, band_id: int) -> None:
        starts = self._burst_starts.get(band_id)
        if not starts:
            return
        pos = bisect_right(starts, slot) - 1        # latest burst starting at or before slot
        if pos < 0:
            return
        start, end = self._bursts_by_band[band_id][pos]
        if start <= slot < end:
            self._intercepted.setdefault((band_id, start), slot)

    # --------------------------------------------------------------- rolling
    @property
    def rolling_detection_rate(self) -> float:
        return float(np.mean(self._rolling)) if self._rolling else 0.0

    @property
    def n_scans(self) -> int:
        return len(self.records)

    # ---------------------------------------------------------------- summary
    def summary(self, change_points: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        n = self.n_scans
        if n == 0:
            return {"scans": 0}

        hits = self.counts[Outcome.HIT]
        misses = self.counts[Outcome.MISS]
        false_alarms = self.counts[Outcome.FALSE_ALARM]
        correct_rejections = self.counts[Outcome.CORRECT_NON_DETECTION]

        active_looks = hits + misses
        inactive_looks = false_alarms + correct_rejections
        pd_measured = hits / active_looks if active_looks else 0.0
        pfa_measured = false_alarms / inactive_looks if inactive_looks else 0.0

        delays = [slot - start for (_, start), slot in self._intercepted.items()]
        total_bursts = self._burst_count
        detections = hits + false_alarms

        p_ml = np.array([r.ml_probability for r in self.records], dtype=float)
        observed = np.array([1.0 if r.outcome.is_detection else 0.0 for r in self.records])
        truth = np.array([1.0 if r.truth_active else 0.0 for r in self.records])

        slots_elapsed = max(1, self.records[-1].slot - self.records[0].slot + 1)
        coverage = float((self.band_scans > 0).mean())
        share = self.band_scans / max(1, self.band_scans.sum())
        nonzero = share[share > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(self.n_bands)) if self.n_bands > 1 else 0.0

        summary: Dict[str, Any] = {
            "scans": n,
            "slots": slots_elapsed,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_non_detections": correct_rejections,

            # --- figures of merit -------------------------------------------
            "probability_of_detection": round(pd_measured, 4),
            "probability_of_false_alarm": round(pfa_measured, 4),
            "sensitivity_index": round(pd_measured - pfa_measured, 4),
            "miss_rate": round(misses / active_looks, 4) if active_looks else 0.0,
            "average_intercept_rate": round(hits / slots_elapsed, 4),
            "intercept_ratio": round(len(self._intercepted) / total_bursts, 4) if total_bursts else 0.0,
            "average_intercept_delay": round(float(np.mean(delays)), 3) if delays else float("nan"),
            "intercept_time_error_rms": round(float(np.sqrt(np.mean(np.square(delays)))), 3)
            if delays else float("nan"),
            "detection_rate": round(hits / n, 4),
            "scan_efficiency": round(detections / n, 4),
            "coverage": round(coverage, 4),
            "coverage_entropy": round(entropy, 4),
            "cumulative_reward": round(self.cumulative_reward, 3),
            "average_reward": round(self.cumulative_reward / n, 4),
            "exploration_share": round(float(np.mean([r.exploration for r in self.records])), 4),

            # --- prediction quality (scored against realised observations) ---
            "percentage_correct_predictions": round(float(np.mean((p_ml >= 0.5) == (observed >= 0.5))), 4),
            "brier_observed": round(float(np.mean((p_ml - observed) ** 2)), 4),
            "brier_truth": round(float(np.mean((p_ml - truth) ** 2)), 4),
            "log_loss_observed": round(float(_log_loss(observed, p_ml)), 4),
            "roc_auc_truth": _safe_auc(truth, p_ml),
            "activity_when_scanned": round(float(truth.mean()), 4),
            "bursts_total": total_bursts,
            "bursts_intercepted": len(self._intercepted),
        }

        adaptation = self.adaptation_times(change_points or getattr(self.environment, "change_points", []))
        if adaptation:
            summary["adaptation"] = adaptation
            finite = [a["adaptation_time"] for a in adaptation if a["adaptation_time"] is not None]
            summary["average_adaptation_time"] = round(float(np.mean(finite)), 2) if finite else None
        return summary

    # ------------------------------------------------------------- adaptation
    def adaptation_times(self, change_points: Sequence[int]) -> List[Dict[str, Any]]:
        """Slots needed after each change point to regain pre-change performance."""
        results: List[Dict[str, Any]] = []
        if not change_points or not self.records:
            return results
        window = self.rolling_window
        slots = np.array([r.slot for r in self.records])
        hits = np.array([1.0 if r.outcome is Outcome.HIT else 0.0 for r in self.records])

        for cp in change_points:
            before = hits[(slots < cp) & (slots >= cp - window)]
            if len(before) < max(10, window // 4):
                continue
            baseline = float(before.mean()) * self.adaptation_tolerance
            after_idx = np.flatnonzero(slots >= cp)
            recovered: Optional[int] = None
            for pos in after_idx:
                lo = max(0, pos - window + 1)
                if slots[pos] - cp >= window // 4 and float(hits[lo:pos + 1].mean()) >= baseline:
                    recovered = int(slots[pos] - cp)
                    break
            results.append({
                "change_point": int(cp),
                "baseline_rate": round(float(before.mean()), 4),
                "target_rate": round(baseline, 4),
                "adaptation_time": recovered,
            })
        return results

    # ------------------------------------------------------------- UI helpers
    def curves(self, max_points: int = 400) -> Dict[str, List[float]]:
        """Down-sampled curves for plotting without shipping every point."""
        def thin(values: List[float]) -> List[float]:
            if len(values) <= max_points:
                return [round(float(v), 4) for v in values]
            step = len(values) / max_points
            return [round(float(values[int(i * step)]), 4) for i in range(max_points)]

        return {
            "cumulative_reward": thin(self._reward_curve),
            "rolling_detection_rate": thin(self._rolling_curve),
        }

    def band_statistics(self) -> Dict[str, List[float]]:
        return {
            "scans": self.band_scans.tolist(),
            "hits": self.band_hits.tolist(),
            "hit_rate": (self.band_hits / np.maximum(1, self.band_scans)).round(4).tolist(),
        }


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, p)), 4)


__all__ = ["EvaluationEngine", "RewardWeights", "StepRecord"]
