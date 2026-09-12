"""Scheduler interface and decision context.

A scheduler converts *what the system believes* into *where to look next*.  It
receives only observation-derived information: ML probabilities, Bayesian
posteriors, observed counts, and the bands the receiver is currently allowed to
tune to.  No scheduler can reach hidden ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Decision


@dataclass(slots=True)
class DecisionContext:
    """Everything a scheduler is permitted to see at one decision point."""

    slot: int
    n_bands: int
    available_bands: List[int]
    rng: np.random.Generator
    ml_probabilities: np.ndarray
    scans: np.ndarray
    detections: np.ndarray
    learner: Any = None                 # BetaBernoulliLearner
    adaptive: Any = None                # AdaptiveLearner or None
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mask(self, values: np.ndarray) -> np.ndarray:
        """Copy of ``values`` with unavailable bands pushed out of contention."""
        out = np.full(self.n_bands, -np.inf, dtype=float)
        idx = np.asarray(self.available_bands, dtype=int)
        out[idx] = np.asarray(values, dtype=float)[idx]
        return out


class Scheduler(ABC):
    """Base class for every scan-selection policy."""

    #: Short key used in configuration, experiments and the UI.
    name: str = "base"
    #: Human-facing label.
    title: str = "Base scheduler"
    #: Whether the policy consumes ML predictions at all.
    uses_ml: bool = False

    def reset(self) -> None:
        """Clear any per-run internal state."""

    @abstractmethod
    def select(self, ctx: DecisionContext) -> Decision:
        """Choose the next band to observe."""

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _first_available(ctx: DecisionContext) -> int:
        if not ctx.available_bands:
            raise RuntimeError(f"no band available to scan at slot {ctx.slot}")
        return int(ctx.available_bands[0])

    @staticmethod
    def _argmax_available(ctx: DecisionContext, values: np.ndarray) -> int:
        masked = ctx.mask(values)
        return int(np.argmax(masked))

    def _decision(self, ctx: DecisionContext, band: int, sampled: float = 0.0,
                  candidates: Optional[Sequence[int]] = None, rationale: str = "",
                  scores: Optional[Dict[int, float]] = None,
                  exploration: bool = False) -> Decision:
        learner = ctx.learner
        if learner is not None:
            state = learner.band_state(band)
            alpha, beta = state["alpha"], state["beta"]
            mean, std = state["mean"], state["std"]
        else:
            hits = float(ctx.detections[band])
            misses = float(ctx.scans[band] - ctx.detections[band])
            alpha, beta = 1.0 + hits, 1.0 + misses
            total = alpha + beta
            mean = alpha / total
            std = float(np.sqrt(alpha * beta / (total * total * (total + 1.0))))
        return Decision(
            slot=ctx.slot,
            band_id=int(band),
            strategy=self.name,
            candidates=[int(c) for c in (candidates if candidates is not None else [band])],
            ml_probability=float(ctx.ml_probabilities[band]),
            posterior_mean=float(mean),
            posterior_std=float(std),
            sampled_value=float(sampled),
            alpha=float(alpha),
            beta=float(beta),
            exploration=bool(exploration),
            adaptive_influence=float(ctx.adaptive.influence()) if ctx.adaptive is not None else 0.0,
            scores={int(k): float(v) for k, v in (scores or {}).items()},
            rationale=rationale,
        )


__all__ = ["Scheduler", "DecisionContext"]
