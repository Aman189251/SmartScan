"""Feature engine.

Turns the observation history into an ML-ready temporal/tabular representation.

Ground-truth barrier: this module is constructed from :class:`NormalizedObservation`
values only.  It has no reference to the environment and no way to reach hidden
activity state, so no feature can leak truth into the live decision path.  The
only place truth is ever attached is offline dataset construction, where it
becomes the training *label* and never a feature.

Every feature is O(1) to update, so a full run costs one array write per slot
rather than a recomputation over history.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Sequence

import numpy as np

FEATURE_VERSION = "fe-1.0"

FEATURE_NAMES: List[str] = [
    "band_norm",              # position of the band in the abstract search space
    "scans_log",              # log1p(times this band has been scanned)
    "detections_log",         # log1p(times a detection was reported)
    "observed_rate",          # smoothed lifetime detection rate
    "rate_short",             # detection rate over the last S observations
    "rate_medium",
    "rate_long",
    "trend_short_long",       # short-window rate minus long-window rate
    "ewma_fast",
    "ewma_slow",
    "ewma_gap",
    "since_scan_norm",        # staleness of this band
    "since_detection_norm",
    "consec_detections",
    "consec_nondetections",
    "p_det_after_det",        # observed transition statistics
    "p_det_after_nondet",
    "scan_share_recent",      # share of recent scanning effort spent here
    "staleness_rank",         # rank of staleness across bands
    "activity_rank",          # rank of recent activity across bands
    "global_rate_recent",     # environment-wide recent detection rate
    "cycle_sin",              # cyclic temporal context
    "cycle_cos",
    "phase_rate",             # detection rate for this band at this cycle phase
    "phase_coverage",         # how well this band/phase cell has been sampled
    "burstiness",             # variability of gaps between detections
    "last_detected",          # outcome of the most recent scan (-1 = never scanned)
    "never_scanned",
]

N_FEATURES = len(FEATURE_NAMES)
_PHASE_BUCKETS = 8
_STALE_CAP = 300.0
_GLOBAL_WINDOW = 200


class FeatureEngine:
    """Maintains per-band observation statistics and emits feature vectors."""

    def __init__(
        self,
        n_bands: int,
        short_window: int = 8,
        medium_window: int = 24,
        long_window: int = 64,
        ewma_fast: float = 0.30,
        ewma_slow: float = 0.05,
        cycle_length: int = 120,
    ):
        self.n_bands = int(n_bands)
        self.short_window = int(short_window)
        self.medium_window = int(medium_window)
        self.long_window = int(long_window)
        self.alpha_fast = float(ewma_fast)
        self.alpha_slow = float(ewma_slow)
        self.cycle_length = max(2, int(cycle_length))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        n = self.n_bands
        self.scans = np.zeros(n, dtype=np.float64)
        self.detections = np.zeros(n, dtype=np.float64)
        self.last_scan = np.full(n, -1, dtype=np.int64)
        self.last_detection = np.full(n, -1, dtype=np.int64)
        self.consec_det = np.zeros(n, dtype=np.float64)
        self.consec_nondet = np.zeros(n, dtype=np.float64)
        self.ewma_fast = np.full(n, 0.5, dtype=np.float64)
        self.ewma_slow = np.full(n, 0.5, dtype=np.float64)

        self.after_det_total = np.zeros(n, dtype=np.float64)
        self.after_det_hits = np.zeros(n, dtype=np.float64)
        self.after_nondet_total = np.zeros(n, dtype=np.float64)
        self.after_nondet_hits = np.zeros(n, dtype=np.float64)
        self.prev_detected = np.full(n, -1, dtype=np.int64)

        self.phase_total = np.zeros((n, _PHASE_BUCKETS), dtype=np.float64)
        self.phase_hits = np.zeros((n, _PHASE_BUCKETS), dtype=np.float64)

        # Three fixed-length windows per band with running sums, so a window
        # rate is a division rather than a re-scan of the history each slot.
        self._win: List[List[Deque[int]]] = [
            [deque(maxlen=self.short_window),
             deque(maxlen=self.medium_window),
             deque(maxlen=self.long_window)]
            for _ in range(n)
        ]
        self._win_sums = np.zeros((n, 3), dtype=np.float64)
        self._detection_slots: List[Deque[int]] = [deque(maxlen=8) for _ in range(n)]

        self._recent_scans: Deque[int] = deque(maxlen=_GLOBAL_WINDOW)
        self._recent_detections: Deque[int] = deque(maxlen=_GLOBAL_WINDOW)
        self._recent_band_counts = np.zeros(n, dtype=np.float64)
        self.total_scans = 0
        self.current_slot = 0

    # ----------------------------------------------------------------- update
    def observe(self, slot: int, band_id: int, detected: bool) -> None:
        """Fold one normalised observation into the running statistics."""
        b = int(band_id)
        d = 1.0 if detected else 0.0
        self.current_slot = int(slot)

        prev = self.prev_detected[b]
        if prev == 1:
            self.after_det_total[b] += 1.0
            self.after_det_hits[b] += d
        elif prev == 0:
            self.after_nondet_total[b] += 1.0
            self.after_nondet_hits[b] += d
        self.prev_detected[b] = int(d)

        self.scans[b] += 1.0
        self.detections[b] += d
        self.last_scan[b] = slot
        if detected:
            self.last_detection[b] = slot
            self.consec_det[b] += 1.0
            self.consec_nondet[b] = 0.0
            self._detection_slots[b].append(int(slot))
        else:
            self.consec_nondet[b] += 1.0
            self.consec_det[b] = 0.0

        self.ewma_fast[b] += self.alpha_fast * (d - self.ewma_fast[b])
        self.ewma_slow[b] += self.alpha_slow * (d - self.ewma_slow[b])

        windows = self._win[b]
        sums = self._win_sums[b]
        for w in range(3):
            buf = windows[w]
            if len(buf) == buf.maxlen:
                sums[w] -= buf[0]
            buf.append(d)
            sums[w] += d

        bucket = self._phase_bucket(slot)
        self.phase_total[b, bucket] += 1.0
        self.phase_hits[b, bucket] += d

        if len(self._recent_scans) == self._recent_scans.maxlen:
            self._recent_band_counts[self._recent_scans[0]] -= 1.0
        self._recent_scans.append(b)
        self._recent_band_counts[b] += 1.0
        self._recent_detections.append(int(d))
        self.total_scans += 1

    def _phase_bucket(self, slot: int) -> int:
        return int((slot % self.cycle_length) * _PHASE_BUCKETS // self.cycle_length)

    # ---------------------------------------------------------------- extract
    def _window_rate(self, band: int, index: int) -> float:
        """Detection rate over window ``index`` (0 short, 1 medium, 2 long)."""
        buf = self._win[band][index]
        if not buf:
            return 0.5
        return float(self._win_sums[band, index] / len(buf))

    def _burstiness(self, band: int) -> float:
        slots = self._detection_slots[band]
        if len(slots) < 3:
            return 0.0
        values = list(slots)
        gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        mean = sum(gaps) / len(gaps)
        if mean <= 0:
            return 0.0
        variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return min(2.0, math.sqrt(variance) / mean)

    def build_matrix(self, slot: int) -> np.ndarray:
        """Feature matrix of shape ``(n_bands, N_FEATURES)`` for decision time ``slot``."""
        n = self.n_bands
        out = np.zeros((n, N_FEATURES), dtype=np.float32)

        since_scan = np.where(self.last_scan >= 0, slot - self.last_scan, _STALE_CAP)
        since_det = np.where(self.last_detection >= 0, slot - self.last_detection, _STALE_CAP)
        stale_rank = _rank_normalised(since_scan)
        activity_rank = _rank_normalised(self.ewma_fast)

        recent_det = self._recent_detections
        global_rate = float(sum(recent_det) / len(recent_det)) if recent_det else 0.5
        recent_total = max(1.0, float(len(self._recent_scans)))
        bucket = self._phase_bucket(slot)
        angle = 2.0 * math.pi * (slot % self.cycle_length) / self.cycle_length
        cycle_sin, cycle_cos = math.sin(angle), math.cos(angle)

        phase_total = self.phase_total[:, bucket]
        phase_hits = self.phase_hits[:, bucket]

        for b in range(n):
            scans = self.scans[b]
            after_det_t = self.after_det_total[b]
            after_non_t = self.after_nondet_total[b]
            row = out[b]
            row[0] = b / max(1, n - 1)
            row[1] = math.log1p(scans)
            row[2] = math.log1p(self.detections[b])
            row[3] = (self.detections[b] + 0.5) / (scans + 1.0)
            row[4] = self._window_rate(b, 0)
            row[5] = self._window_rate(b, 1)
            row[6] = self._window_rate(b, 2)
            row[7] = row[4] - row[6]
            row[8] = self.ewma_fast[b]
            row[9] = self.ewma_slow[b]
            row[10] = self.ewma_fast[b] - self.ewma_slow[b]
            row[11] = min(1.0, since_scan[b] / _STALE_CAP)
            row[12] = min(1.0, since_det[b] / _STALE_CAP)
            row[13] = min(1.0, self.consec_det[b] / 10.0)
            row[14] = min(1.0, self.consec_nondet[b] / 20.0)
            row[15] = (self.after_det_hits[b] + 0.5) / (after_det_t + 1.0)
            row[16] = (self.after_nondet_hits[b] + 0.5) / (after_non_t + 1.0)
            row[17] = self._recent_band_counts[b] / recent_total
            row[18] = stale_rank[b]
            row[19] = activity_rank[b]
            row[20] = global_rate
            row[21] = cycle_sin
            row[22] = cycle_cos
            row[23] = (phase_hits[b] + 0.5) / (phase_total[b] + 1.0)
            row[24] = min(1.0, phase_total[b] / 8.0)
            row[25] = self._burstiness(b)
            row[26] = float(self.prev_detected[b])
            row[27] = 1.0 if scans == 0 else 0.0
        return out

    def build_vector(self, slot: int, band_id: int) -> np.ndarray:
        return self.build_matrix(slot)[band_id]

    # ------------------------------------------------------------- inspection
    def summary(self) -> Dict[str, object]:
        return {
            "feature_version": FEATURE_VERSION,
            "n_features": N_FEATURES,
            "n_bands": self.n_bands,
            "total_scans": self.total_scans,
            "coverage": float((self.scans > 0).mean()),
            "observed_rate": float(self.detections.sum() / max(1.0, self.scans.sum())),
        }


def _rank_normalised(values: np.ndarray) -> np.ndarray:
    """Rank of each entry in [0, 1]; ties broken by position, cheap and stable."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    denom = max(1, len(values) - 1)
    return ranks / denom


def feature_frame_columns() -> Sequence[str]:
    return tuple(FEATURE_NAMES)


__all__ = [
    "FeatureEngine",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "N_FEATURES",
    "feature_frame_columns",
]
