"""Hidden adaptive learning layer.

Runs continuously in the background beside the main decision path.  It keeps its
own fast-forgetting belief over the bands, scores itself against the offline
model on every executed scan, watches for environment change, and is only
allowed to influence the decision layer once measured acceptance criteria have
been met for a sustained period.

State ladder (matching the design document):

    OFFLINE_PRIOR -> LEARNING -> SHADOW -> ADAPTIVE_ASSISTED -> ADAPTIVE_CONTROL

Promotion is never automatic on a slot count alone.  It requires recent
predictive quality (Brier), calibration error, a measured advantage over the
offline model, and stability across consecutive evaluations.  A drift alarm or
a failed criterion demotes the layer immediately.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..core.types import AdaptiveState
from .bayesian import BetaBernoulliLearner


class PageHinkley:
    """Page-Hinkley change detector over a stream of prediction errors.

    Tracks the cumulative deviation of the error from its running mean and
    alarms when the drop from the running minimum exceeds ``threshold``.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 0.60, min_samples: int = 40):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, error: float) -> bool:
        """Feed one error value; returns True when a change point is flagged."""
        self.n += 1
        self.mean += (error - self.mean) / self.n
        self.cumulative += error - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        if self.n < self.min_samples:
            return False
        if self.cumulative - self.minimum > self.threshold:
            self.reset()
            return True
        return False


class WindowedDriftDetector:
    """Two-window test: is recent predictive error materially worse than baseline?

    Page-Hinkley accumulates every deviation, which makes it sensitive to the
    slow, self-inflicted drift this system generates on its own: as the
    scheduler concentrates on productive bands, the distribution of what gets
    scanned changes constantly, and a cumulative detector reads that as change.

    This test instead compares the mean error over the most recent ``window``
    scans against the ``reference`` scans before them, and alarms only when the
    gap exceeds ``z_threshold`` standard errors.  The threshold is therefore in
    interpretable units, and gradual self-inflicted drift moves both windows
    together and cancels out.
    """

    def __init__(self, window: int = 60, reference: int = 240,
                 z_threshold: float = 3.0, min_gap: float = 0.05):
        self.window = int(window)
        self.reference = int(reference)
        self.z_threshold = float(z_threshold)
        self.min_gap = float(min_gap)
        self.reset()

    def reset(self) -> None:
        self._buffer: Deque[float] = deque(maxlen=self.window + self.reference)
        self.last_z = 0.0
        self.last_gap = 0.0

    def update(self, error: float) -> bool:
        self._buffer.append(float(error))
        if len(self._buffer) < self._buffer.maxlen:
            return False

        values = np.fromiter(self._buffer, dtype=float)
        recent = values[-self.window:]
        baseline = values[:-self.window]

        gap = float(recent.mean() - baseline.mean())
        spread = float(baseline.std())
        standard_error = max(1e-6, spread * math.sqrt(1.0 / self.window + 1.0 / len(baseline)))
        self.last_gap = gap
        self.last_z = gap / standard_error

        if gap >= self.min_gap and self.last_z >= self.z_threshold:
            self.reset()
            return True
        return False


class ScoreMonitor:
    """Sliding-window predictive quality over executed scans."""

    def __init__(self, window: int = 120, n_bins: int = 10):
        self.window = int(window)
        self.n_bins = int(n_bins)
        self._pred: Deque[float] = deque(maxlen=self.window)
        self._obs: Deque[int] = deque(maxlen=self.window)
        self.total = 0

    def add(self, probability: float, detected: bool) -> None:
        self._pred.append(float(np.clip(probability, 1e-6, 1 - 1e-6)))
        self._obs.append(int(bool(detected)))
        self.total += 1

    def __len__(self) -> int:
        return len(self._pred)

    @property
    def brier(self) -> float:
        if not self._pred:
            return 0.25
        p = np.fromiter(self._pred, dtype=float)
        y = np.fromiter(self._obs, dtype=float)
        return float(np.mean((p - y) ** 2))

    @property
    def log_loss(self) -> float:
        if not self._pred:
            return math.log(2.0)
        p = np.fromiter(self._pred, dtype=float)
        y = np.fromiter(self._obs, dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    @property
    def accuracy(self) -> float:
        """Percentage of correct predictions at the 0.5 decision threshold."""
        if not self._pred:
            return 0.0
        p = np.fromiter(self._pred, dtype=float)
        y = np.fromiter(self._obs, dtype=float)
        return float(np.mean((p >= 0.5) == (y >= 0.5)))

    @property
    def calibration_error(self) -> float:
        """Expected calibration error over equal-width probability bins."""
        if len(self._pred) < 10:
            return 1.0
        p = np.fromiter(self._pred, dtype=float)
        y = np.fromiter(self._obs, dtype=float)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, self.n_bins - 1)
        error = 0.0
        for b in range(self.n_bins):
            mask = idx == b
            if not mask.any():
                continue
            error += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
        return float(error)

    def summary(self) -> Dict[str, float]:
        return {
            "samples": len(self._pred),
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "calibration_error": round(self.calibration_error, 4),
        }


@dataclass(slots=True)
class AdaptiveEvent:
    slot: int
    event_type: str
    detail: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "event_type": self.event_type,
            "detail": self.detail,
            "payload": dict(self.payload),
        }


class AdaptiveLearner:
    """Background learner, monitor, drift detector and promotion controller."""

    def __init__(self, n_bands: int, cfg, enabled: bool = True):
        acceptance = cfg.get_path("adaptive.acceptance", {}) or {}
        acc = acceptance.to_dict() if hasattr(acceptance, "to_dict") else dict(acceptance)

        self.n_bands = int(n_bands)
        self.enabled = bool(enabled)
        self.window = int(cfg.get_path("adaptive.window", 120))
        self.max_influence = float(cfg.get_path("adaptive.max_influence", 0.5))
        self.drift_cooldown = int(cfg.get_path("adaptive.drift_cooldown", 150))
        self.drift_decay = float(cfg.get_path("bayesian.drift_decay_lambda", 0.90))
        self.smoothing = max(2, int(cfg.get_path("adaptive.drift_smoothing", 25)))

        self.min_observations = int(acc.get("min_observations", 200))
        self.max_brier = float(acc.get("max_brier", 0.20))
        self.max_calibration_error = float(acc.get("max_calibration_error", 0.10))
        self.min_agreement = float(acc.get("min_agreement", 0.55))
        self.stability_window = int(acc.get("stability_window", 3))

        # The adaptive layer keeps its own faster-forgetting belief so it can
        # disagree with the main path. This is what "shadow mode" compares.
        self.shadow = BetaBernoulliLearner(
            n_bands=n_bands,
            prior_strength_k=float(cfg.get_path("bayesian.prior_strength_k", 10.0)) * 0.5,
            decay_lambda=min(0.99, float(cfg.get_path("bayesian.decay_lambda", 0.995))),
            prior_mode="dynamic",
        )
        self.drift_signal = str(cfg.get_path("adaptive.drift_signal", "belief_error"))
        self.drift_min_evidence = float(cfg.get_path("adaptive.drift_min_evidence", 4.0))
        self.detector_kind = str(cfg.get_path("adaptive.drift_detector", "windowed"))
        if self.detector_kind == "page_hinkley":
            self.detector = PageHinkley(
                delta=float(cfg.get_path("adaptive.page_hinkley_delta", 0.01)),
                threshold=float(cfg.get_path("adaptive.page_hinkley_lambda", 2.0)),
            )
        else:
            self.detector = WindowedDriftDetector(
                window=int(cfg.get_path("adaptive.drift_window", 60)),
                reference=int(cfg.get_path("adaptive.drift_reference", 240)),
                z_threshold=float(cfg.get_path("adaptive.drift_z_threshold", 3.0)),
                min_gap=float(cfg.get_path("adaptive.drift_min_gap", 0.05)),
            )
        self.offline_monitor = ScoreMonitor(self.window)
        self.adaptive_monitor = ScoreMonitor(self.window)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.state = AdaptiveState.OFFLINE_PRIOR
        self.shadow.reset()
        self.detector.reset()
        self.offline_monitor = ScoreMonitor(self.window)
        self.adaptive_monitor = ScoreMonitor(self.window)
        self._advantage: Deque[int] = deque(maxlen=self.window)
        # Per-scan squared error is a Bernoulli outcome and far too noisy to feed
        # a change detector directly, so the detector sees a smoothed error.
        self._error_window: Deque[float] = deque(maxlen=self.smoothing)
        self._pass_streak = 0
        self._observations = 0
        self._drift_until = -1
        self.drift_flag = False
        self.last_drift_slot: Optional[int] = None
        self.drift_count = 0
        self.events: List[AdaptiveEvent] = []
        self.recommendation: Optional[int] = None

    # ------------------------------------------------------------- prediction
    def shadow_probabilities(self, p_ml: np.ndarray) -> np.ndarray:
        """The adaptive layer's own activity estimate for every band."""
        self.shadow.set_prior(np.asarray(p_ml, dtype=float))
        return self.shadow.mean

    def recommend(self, p_ml: np.ndarray, rng: np.random.Generator) -> int:
        """Parallel recommendation. In SHADOW this is recorded, never executed."""
        samples = self.shadow.sample(rng)
        self.recommendation = int(np.argmax(samples))
        return self.recommendation

    def influence(self) -> float:
        """Weight the decision layer may give to the adaptive estimate."""
        if not self.enabled:
            return 0.0
        if self.state is AdaptiveState.ADAPTIVE_ASSISTED:
            return self.max_influence
        if self.state is AdaptiveState.ADAPTIVE_CONTROL:
            return 1.0
        return 0.0

    # ---------------------------------------------------------------- observe
    def observe(self, slot: int, band_id: int, detected: bool,
                p_offline: float, p_adaptive: float) -> List[AdaptiveEvent]:
        """Fold one executed observation into every background monitor."""
        new_events: List[AdaptiveEvent] = []
        if not self.enabled:
            return new_events

        self._observations += 1
        evidence_before = float(self.shadow.evidence[band_id])
        self.shadow.update(band_id, detected)
        self.shadow.decay()

        y = 1.0 if detected else 0.0
        err_offline = (p_offline - y) ** 2
        err_adaptive = (p_adaptive - y) ** 2
        self.offline_monitor.add(p_offline, detected)
        self.adaptive_monitor.add(p_adaptive, detected)
        self._advantage.append(1 if err_adaptive <= err_offline else 0)
        self._error_window.append(err_offline)

        if self.drift_flag:
            # Already adapting: hold the alarm until the cooldown expires, then
            # re-baseline the detector against the new regime.
            if slot >= self._drift_until:
                self.drift_flag = False
                self.shadow.restore_decay()
                self.detector.reset()
                self._error_window.clear()
                new_events.append(AdaptiveEvent(
                    slot, "DRIFT_CLEARED", "evidence decay returned to baseline"))
        else:
            signal = self._drift_signal(err_offline, err_adaptive, evidence_before)
            if signal is not None and self._feed_detector(signal):
                new_events.append(self._raise_drift(slot))

        promotion = self._evaluate_acceptance(slot)
        if promotion is not None:
            new_events.append(promotion)

        self.events.extend(new_events)
        return new_events

    def _drift_signal(self, err_offline: float, err_belief: float,
                      evidence: float) -> Optional[float]:
        """Choose what the change detector actually watches.

        ``model_error`` watches the offline model's error on every executed scan.
        That stream is non-stationary by construction: as the scheduler
        concentrates on productive bands, what gets scanned keeps changing, and a
        detector reads the policy's own improvement as environmental change.

        ``belief_error`` (the default) watches only looks at bands the system was
        already confident about, and scores them against the *belief* rather than
        the offline model. Exploring an unknown band no longer registers, so what
        remains is closer to the question that matters: is what we confidently
        believed still true.  Returns None when a scan carries no drift evidence.
        """
        if self.drift_signal == "model_error":
            return err_offline
        if evidence < self.drift_min_evidence:
            return None
        return err_belief

    def _feed_detector(self, error: float) -> bool:
        """Hand one prediction error to the configured change detector.

        The windowed test averages internally, so it takes the raw error.
        Page-Hinkley accumulates every sample and needs the error smoothed
        first, or Bernoulli noise alone will trip it.
        """
        if self.detector_kind != "page_hinkley":
            return bool(self.detector.update(error))
        if len(self._error_window) < self._error_window.maxlen:
            return False
        return bool(self.detector.update(sum(self._error_window) / len(self._error_window)))

    def _raise_drift(self, slot: int) -> AdaptiveEvent:
        self.drift_flag = True
        self.drift_count += 1
        self.last_drift_slot = slot
        self._drift_until = slot + self.drift_cooldown
        self.shadow.set_decay(self.drift_decay)
        self._pass_streak = 0
        demoted = False
        if self.state in (AdaptiveState.ADAPTIVE_ASSISTED, AdaptiveState.ADAPTIVE_CONTROL):
            self.state = AdaptiveState.SHADOW
            demoted = True
        return AdaptiveEvent(
            slot,
            "DRIFT_DETECTED",
            "recent observations contradict confident beliefs; stale evidence is being "
            "discounted" + (" and adaptive influence withdrawn" if demoted else "")
            + ". This flags degraded agreement, not a confirmed environment change",
            {
                "brier": self.offline_monitor.brier,
                "cooldown_until": self._drift_until,
                "demoted": demoted,
            },
        )

    # ------------------------------------------------------------- acceptance
    def criteria(self) -> Dict[str, Any]:
        adv = float(np.mean(self._advantage)) if self._advantage else 0.0
        checks = {
            "observations": (self._observations, self.min_observations,
                             self._observations >= self.min_observations),
            "brier": (round(self.adaptive_monitor.brier, 4), self.max_brier,
                      self.adaptive_monitor.brier <= self.max_brier),
            "calibration_error": (round(self.adaptive_monitor.calibration_error, 4),
                                  self.max_calibration_error,
                                  self.adaptive_monitor.calibration_error <= self.max_calibration_error),
            "advantage": (round(adv, 4), self.min_agreement, adv >= self.min_agreement),
            "no_drift": (not self.drift_flag, True, not self.drift_flag),
        }
        return {
            "checks": {k: {"value": v[0], "threshold": v[1], "passed": bool(v[2])}
                       for k, v in checks.items()},
            "all_passed": all(v[2] for v in checks.values()),
            "streak": self._pass_streak,
            "stability_window": self.stability_window,
        }

    def _evaluate_acceptance(self, slot: int) -> Optional[AdaptiveEvent]:
        status = self.criteria()

        if self.state is AdaptiveState.OFFLINE_PRIOR and self._observations >= 1:
            self.state = AdaptiveState.LEARNING
            return AdaptiveEvent(slot, "STATE_CHANGE",
                                 "live observations are now updating the Bayesian state",
                                 {"state": self.state.value})

        if self.state is AdaptiveState.LEARNING and self._observations >= max(20, self.window // 4):
            self.state = AdaptiveState.SHADOW
            return AdaptiveEvent(slot, "STATE_CHANGE",
                                 "adaptive learner is producing parallel recommendations "
                                 "without controlling the scheduler",
                                 {"state": self.state.value})

        if status["all_passed"]:
            self._pass_streak += 1
        else:
            if self._pass_streak > 0 and self.state in (AdaptiveState.ADAPTIVE_ASSISTED,
                                                        AdaptiveState.ADAPTIVE_CONTROL):
                self._pass_streak = 0
                self.state = AdaptiveState.SHADOW
                return AdaptiveEvent(slot, "STATE_CHANGE",
                                     "acceptance criteria no longer met; influence withdrawn",
                                     {"state": self.state.value, "criteria": status["checks"]})
            self._pass_streak = 0
            return None

        if self._pass_streak < self.stability_window:
            return None

        if self.state is AdaptiveState.SHADOW:
            self.state = AdaptiveState.ADAPTIVE_ASSISTED
            self._pass_streak = 0
            return AdaptiveEvent(slot, "STATE_CHANGE",
                                 f"acceptance criteria met; adaptive information may influence "
                                 f"the decision layer up to {self.max_influence:.0%}",
                                 {"state": self.state.value, "criteria": status["checks"]})

        if self.state is AdaptiveState.ADAPTIVE_ASSISTED and self._observations >= self.min_observations * 3:
            self.state = AdaptiveState.ADAPTIVE_CONTROL
            self._pass_streak = 0
            return AdaptiveEvent(slot, "STATE_CHANGE",
                                 "sustained acceptance; adaptive state is providing the "
                                 "approved level of control to the decision layer",
                                 {"state": self.state.value, "criteria": status["checks"]})
        return None

    # ------------------------------------------------------------- inspection
    def status(self) -> Dict[str, Any]:
        adv = float(np.mean(self._advantage)) if self._advantage else 0.0
        return {
            "enabled": self.enabled,
            "state": self.state.value,
            "influence": round(self.influence(), 3),
            "observations": self._observations,
            "drift_flag": self.drift_flag,
            "drift_count": self.drift_count,
            "last_drift_slot": self.last_drift_slot,
            "recommendation": self.recommendation,
            "advantage": round(adv, 4),
            "offline": self.offline_monitor.summary(),
            "adaptive": self.adaptive_monitor.summary(),
            "acceptance": self.criteria(),
        }

    def drain_events(self) -> List[AdaptiveEvent]:
        events, self.events = self.events, []
        return events


__all__ = ["AdaptiveLearner", "AdaptiveEvent", "PageHinkley", "WindowedDriftDetector",
           "ScoreMonitor"]
