"""Shared value types for the Smart Scan intelligence core.

These types form the contract between the receiver boundary, the intelligence
layers and persistence.  Nothing here knows about hardware: the same
``NormalizedObservation`` is produced by the simulator, by a recorded-data
replay, or by an external laboratory receiver adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Outcome(str, Enum):
    """Normalised result of observing one band during one scan."""

    HIT = "HIT"                              # activity present and detected
    MISS = "MISS"                            # activity present, not detected
    FALSE_ALARM = "FALSE_ALARM"              # no activity, detection reported
    CORRECT_NON_DETECTION = "CORRECT_NON_DETECTION"  # no activity, nothing reported

    @property
    def is_detection(self) -> bool:
        """True when the receiver reported energy, regardless of truth."""
        return self in (Outcome.HIT, Outcome.FALSE_ALARM)

    @property
    def is_true_positive(self) -> bool:
        return self is Outcome.HIT


class AdaptiveState(str, Enum):
    """Controlled promotion ladder for the hidden adaptive learner."""

    OFFLINE_PRIOR = "OFFLINE_PRIOR"
    LEARNING = "LEARNING"
    SHADOW = "SHADOW"
    ADAPTIVE_ASSISTED = "ADAPTIVE_ASSISTED"
    ADAPTIVE_CONTROL = "ADAPTIVE_CONTROL"


class RunState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass(slots=True)
class ScanCommand:
    """A validated instruction handed to the receiver adapter."""

    run_id: str
    slot: int
    band_id: int
    dwell: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedObservation:
    """Common observation schema consumed by the intelligence engine."""

    run_id: str
    source_id: str
    slot: int
    timestamp: float
    band_id: int
    scan_duration: int
    outcome: Outcome
    detected: bool
    quality: float = 1.0
    receiver_state: str = "READY"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "slot": self.slot,
            "timestamp": self.timestamp,
            "band_id": self.band_id,
            "scan_duration": self.scan_duration,
            "outcome": self.outcome.value,
            "detected": self.detected,
            "quality": self.quality,
            "receiver_state": self.receiver_state,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Decision:
    """One scheduling decision, carrying everything needed to explain it."""

    slot: int
    band_id: int
    strategy: str
    candidates: List[int] = field(default_factory=list)
    ml_probability: float = 0.0
    posterior_mean: float = 0.0
    posterior_std: float = 0.0
    sampled_value: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    exploration: bool = False
    adaptive_influence: float = 0.0
    scores: Dict[int, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "band_id": self.band_id,
            "strategy": self.strategy,
            "candidates": list(self.candidates),
            "ml_probability": self.ml_probability,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "sampled_value": self.sampled_value,
            "alpha": self.alpha,
            "beta": self.beta,
            "exploration": self.exploration,
            "adaptive_influence": self.adaptive_influence,
            "scores": {int(k): float(v) for k, v in self.scores.items()},
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class StepResult:
    """Full snapshot of one decision cycle, used by the UI and by persistence."""

    slot: int
    decision: Decision
    observation: NormalizedObservation
    reward: float
    truth_active: Optional[bool]
    ml_probabilities: List[float]
    posterior_means: List[float]
    posterior_stds: List[float]
    adaptive_state: AdaptiveState
    drift_flag: bool
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "decision": self.decision.to_dict(),
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "truth_active": self.truth_active,
            "adaptive_state": self.adaptive_state.value,
            "drift_flag": self.drift_flag,
            "metrics": dict(self.metrics),
        }


def band_label(band_id: int) -> str:
    """Human-facing abstract band label, e.g. 0 -> 'B01'."""
    return f"B{band_id + 1:02d}"


__all__ = [
    "Outcome",
    "AdaptiveState",
    "RunState",
    "ScanCommand",
    "NormalizedObservation",
    "Decision",
    "StepResult",
    "band_label",
]
