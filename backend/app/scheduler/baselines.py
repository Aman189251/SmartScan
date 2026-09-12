"""Baseline schedulers.

These exist so the Smart Scan strategy is justified by measurement rather than
assertion.  Every one of them runs against the identical seeded environment and
receiver conditions used by the full strategy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.types import Decision, band_label
from .base import DecisionContext, Scheduler


class RandomScheduler(Scheduler):
    """Control baseline: uniform random band selection."""

    name = "random"
    title = "Random"

    def select(self, ctx: DecisionContext) -> Decision:
        band = int(ctx.rng.choice(ctx.available_bands))
        return self._decision(
            ctx, band,
            sampled=float(ctx.rng.random()),
            candidates=ctx.available_bands,
            rationale=f"Uniform random choice over {len(ctx.available_bands)} available bands.",
            exploration=True,
        )


class RoundRobinScheduler(Scheduler):
    """Deterministic conventional sweep: the classic open-loop strategy."""

    name = "round_robin"
    title = "Round-Robin"

    def __init__(self) -> None:
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def select(self, ctx: DecisionContext) -> Decision:
        available = set(ctx.available_bands)
        band = None
        for _ in range(ctx.n_bands):
            candidate = self._cursor % ctx.n_bands
            self._cursor += 1
            if candidate in available:
                band = candidate
                break
        if band is None:
            band = self._first_available(ctx)
        return self._decision(
            ctx, band,
            candidates=[band],
            rationale=f"Fixed cyclic sweep position {band + 1} of {ctx.n_bands}.",
        )


class GreedyMLScheduler(Scheduler):
    """Always take the highest offline ML probability. No exploration."""

    name = "greedy_ml"
    title = "Greedy ML"
    uses_ml = True

    def select(self, ctx: DecisionContext) -> Decision:
        band = self._argmax_available(ctx, ctx.ml_probabilities)
        scores: Dict[int, float] = {
            int(b): float(ctx.ml_probabilities[b])
            for b in np.argsort(ctx.mask(ctx.ml_probabilities))[::-1][:5]
        }
        return self._decision(
            ctx, band,
            sampled=float(ctx.ml_probabilities[band]),
            candidates=list(scores.keys()),
            scores=scores,
            rationale=(f"Highest offline model probability: {band_label(band)} at "
                       f"{ctx.ml_probabilities[band]:.2f}."),
        )


class UCBScheduler(Scheduler):
    """Upper-confidence-bound baseline over observed detection rates."""

    name = "ucb"
    title = "UCB"

    def __init__(self, c: float = 1.4):
        self.c = float(c)

    def select(self, ctx: DecisionContext) -> Decision:
        counts = np.maximum(ctx.scans, 1e-9)
        means = ctx.detections / counts
        total = max(1.0, float(ctx.scans.sum()))
        bonus = self.c * np.sqrt(2.0 * np.log(total + 1.0) / counts)
        # An unvisited band gets an unbounded bonus, which forces initial coverage.
        scores = np.where(ctx.scans > 0, means + bonus, np.inf)
        band = self._argmax_available(ctx, scores)
        exploration = bool(ctx.scans[band] == 0 or bonus[band] > means[band])
        top = np.argsort(ctx.mask(np.where(np.isfinite(scores), scores, 1e6)))[::-1][:5]
        return self._decision(
            ctx, band,
            sampled=float(means[band] + (bonus[band] if np.isfinite(bonus[band]) else 0.0)),
            candidates=[int(b) for b in top],
            scores={int(b): float(means[b] + min(bonus[b], 9.9)) for b in top},
            rationale=(f"Observed rate {means[band]:.2f} plus exploration bonus "
                       f"{min(bonus[band], 9.9):.2f} on {ctx.scans[band]:.0f} prior scans."),
            exploration=exploration,
        )


class ThompsonScheduler(Scheduler):
    """Thompson Sampling over observation-only Beta posteriors (uniform prior).

    This isolates the mathematical decision layer from the offline model, so an
    ablation can show what the ML prior actually contributes.
    """

    name = "thompson"
    title = "Thompson Sampling"

    def select(self, ctx: DecisionContext) -> Decision:
        alpha = 1.0 + ctx.detections
        beta = 1.0 + (ctx.scans - ctx.detections)
        samples = ctx.rng.beta(alpha, beta)
        band = self._argmax_available(ctx, samples)
        top = np.argsort(ctx.mask(samples))[::-1][:5]
        total = alpha[band] + beta[band]
        mean = alpha[band] / total
        std = float(np.sqrt(alpha[band] * beta[band] / (total * total * (total + 1.0))))
        decision = self._decision(
            ctx, band,
            sampled=float(samples[band]),
            candidates=[int(b) for b in top],
            scores={int(b): float(samples[b]) for b in top},
            rationale=(f"Thompson draw {samples[band]:.2f} from Beta({alpha[band]:.1f}, "
                       f"{beta[band]:.1f}) built from observations alone."),
            exploration=bool(samples[band] > mean + std),
        )
        decision.alpha = float(alpha[band])
        decision.beta = float(beta[band])
        decision.posterior_mean = float(mean)
        decision.posterior_std = std
        return decision


__all__ = [
    "RandomScheduler",
    "RoundRobinScheduler",
    "GreedyMLScheduler",
    "UCBScheduler",
    "ThompsonScheduler",
]
