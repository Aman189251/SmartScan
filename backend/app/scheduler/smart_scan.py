"""The Smart Scan strategy: offline prediction, mathematical decision, execution.

One decision cycle:

1. Take the offline model's activity probability for every band.
2. Blend in the adaptive layer's estimate, but only up to the influence weight
   the adaptive layer has actually earned (zero while it is in shadow mode).
3. Use that blended probability to anchor each band's Beta prior.
4. Build a candidate set from the most promising bands, plus a coverage guard so
   a band can never be starved indefinitely.
5. Draw one Thompson sample per candidate from its posterior and take the
   highest draw.

The offline model never selects a band by itself.  It supplies prior knowledge;
evidence adapts it; the mathematical layer converts belief and uncertainty into
the action.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.types import Decision, band_label
from .base import DecisionContext, Scheduler


class SmartScanScheduler(Scheduler):
    """Offline ML prior + Beta-Bernoulli belief + Thompson Sampling."""

    name = "smart_scan"
    title = "Smart Scan"
    uses_ml = True

    def __init__(self, candidate_k: int = 5, exploration_floor: float = 0.02,
                 coverage_guard: bool = True):
        self.candidate_k = max(1, int(candidate_k))
        self.exploration_floor = float(exploration_floor)
        self.coverage_guard = bool(coverage_guard)

    # ------------------------------------------------------------------ select
    def select(self, ctx: DecisionContext) -> Decision:
        available = list(ctx.available_bands)
        if not available:
            raise RuntimeError(f"no band available to scan at slot {ctx.slot}")

        p_ml = np.asarray(ctx.ml_probabilities, dtype=float)
        influence = 0.0
        p_effective = p_ml
        if ctx.adaptive is not None and ctx.adaptive.enabled:
            influence = float(ctx.adaptive.influence())
            if influence > 0.0:
                p_shadow = np.asarray(ctx.adaptive.shadow_probabilities(p_ml), dtype=float)
                p_effective = (1.0 - influence) * p_ml + influence * p_shadow

        # Layer 2: the ML probability anchors the prior, evidence moves it.
        learner = ctx.learner
        learner.set_prior(p_effective)
        posterior_mean = learner.mean
        posterior_std = learner.std

        # Forced exploration keeps a floor under coverage regardless of belief.
        if self.exploration_floor > 0.0 and ctx.rng.random() < self.exploration_floor:
            band = int(ctx.rng.choice(available))
            samples = learner.sample(ctx.rng)
            return self._build(ctx, band, samples, [band], p_effective, influence,
                               exploration=True,
                               rationale=("Scheduled exploration draw: belief is deliberately "
                                          "set aside to maintain search coverage."))

        candidates = self._candidate_set(ctx, p_effective, available)
        samples = learner.sample(ctx.rng)
        masked = np.full(ctx.n_bands, -np.inf)
        masked[candidates] = samples[candidates]
        band = int(np.argmax(masked))

        exploration = bool(samples[band] > posterior_mean[band] + posterior_std[band])
        rationale = (
            f"{band_label(band)} produced the highest Thompson draw "
            f"({samples[band]:.2f}) among {len(candidates)} candidates. "
            f"Model probability {p_ml[band]:.2f}, current belief {posterior_mean[band]:.2f} "
            f"+/- {posterior_std[band]:.2f}."
        )
        if influence > 0.0:
            rationale += f" Adaptive layer influence {influence:.0%}."
        if exploration:
            rationale += " The draw exceeded the posterior mean, so this is an exploratory look."
        return self._build(ctx, band, samples, candidates, p_effective, influence,
                           exploration=exploration, rationale=rationale)

    # -------------------------------------------------------------- candidates
    def _candidate_set(self, ctx: DecisionContext, p_effective: np.ndarray,
                       available: List[int]) -> List[int]:
        avail = np.asarray(available, dtype=int)
        k = min(self.candidate_k, len(avail))
        ranked = avail[np.argsort(p_effective[avail])[::-1]]
        candidates = [int(b) for b in ranked[:k]]

        if self.coverage_guard:
            since_scan = ctx.metadata.get("since_scan")
            if since_scan is not None:
                stalest = int(avail[int(np.argmax(np.asarray(since_scan)[avail]))])
                if stalest not in candidates:
                    candidates.append(stalest)
        return candidates

    def _build(self, ctx: DecisionContext, band: int, samples: np.ndarray,
               candidates: List[int], p_effective: np.ndarray, influence: float,
               exploration: bool, rationale: str) -> Decision:
        scores: Dict[int, float] = {int(b): float(samples[b]) for b in candidates}
        decision = self._decision(
            ctx, band,
            sampled=float(samples[band]),
            candidates=candidates,
            scores=scores,
            rationale=rationale,
            exploration=exploration,
        )
        decision.adaptive_influence = influence
        decision.ml_probability = float(ctx.ml_probabilities[band])
        return decision


__all__ = ["SmartScanScheduler"]
