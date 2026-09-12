"""Beta-Bernoulli belief engine.

Each abstract band carries a Beta distribution over its current activity
probability.  The offline model supplies the prior; live observations supply the
likelihood:

    alpha_b = k * p_ML(b) + H_b
    beta_b  = k * (1 - p_ML(b)) + M_b

``H_b`` and ``M_b`` are decayed counts of detections and non-detections, so old
evidence loses influence at a controlled rate instead of dominating for ever:

    H_t = lambda * H_(t-1) + hit_t,    M_t = lambda * M_(t-1) + miss_t

Two prior modes are supported.  ``dynamic`` re-anchors the prior on the current
ML probability at every decision, which lets a fresh prediction move belief
immediately.  ``static`` fixes the prior at first contact, matching the simpler
formulation in the design document.  Both keep the full posterior, so the
scheduler always has uncertainty available, not just a point estimate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_EPS = 1e-3


class BetaBernoulliLearner:
    """Per-band Beta posteriors with exponential evidence decay."""

    def __init__(
        self,
        n_bands: int,
        prior_strength_k: float = 10.0,
        decay_lambda: float = 0.995,
        prior_mode: str = "dynamic",
        min_evidence: float = 0.0,
    ):
        if prior_mode not in ("dynamic", "static"):
            raise ValueError("prior_mode must be 'dynamic' or 'static'")
        if not 0.0 < decay_lambda <= 1.0:
            raise ValueError("decay_lambda must lie in (0, 1]")
        self.n_bands = int(n_bands)
        self.k = float(prior_strength_k)
        self.base_decay = float(decay_lambda)
        self.decay_lambda = float(decay_lambda)
        self.prior_mode = prior_mode
        self.min_evidence = float(min_evidence)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        n = self.n_bands
        self.hits = np.zeros(n, dtype=np.float64)
        self.misses = np.zeros(n, dtype=np.float64)
        self.prior_alpha = np.full(n, self.k * 0.5, dtype=np.float64)
        self.prior_beta = np.full(n, self.k * 0.5, dtype=np.float64)
        self._prior_set = np.zeros(n, dtype=bool)
        self.updates = 0
        self.decay_lambda = self.base_decay

    # ------------------------------------------------------------------ prior
    def set_prior(self, p_ml: np.ndarray) -> None:
        """Anchor priors on ML probabilities.

        In ``static`` mode a band's prior is written once, the first time a
        prediction is seen; afterwards only evidence moves the posterior.
        """
        p = np.clip(np.asarray(p_ml, dtype=np.float64), _EPS, 1.0 - _EPS)
        if self.prior_mode == "dynamic":
            self.prior_alpha = self.k * p
            self.prior_beta = self.k * (1.0 - p)
            self._prior_set[:] = True
        else:
            fresh = ~self._prior_set
            if fresh.any():
                self.prior_alpha[fresh] = self.k * p[fresh]
                self.prior_beta[fresh] = self.k * (1.0 - p[fresh])
                self._prior_set[fresh] = True

    # ----------------------------------------------------------------- update
    def update(self, band_id: int, detected: bool) -> None:
        """Fold one observation into the evidence counts."""
        if detected:
            self.hits[band_id] += 1.0
        else:
            self.misses[band_id] += 1.0
        self.updates += 1

    def decay(self, lam: Optional[float] = None) -> None:
        """Apply one step of exponential forgetting to all evidence."""
        factor = self.decay_lambda if lam is None else float(lam)
        if factor >= 1.0:
            return
        self.hits *= factor
        self.misses *= factor
        if self.min_evidence > 0.0:
            np.maximum(self.hits, self.min_evidence, out=self.hits)
            np.maximum(self.misses, self.min_evidence, out=self.misses)

    def set_decay(self, lam: float) -> None:
        self.decay_lambda = float(np.clip(lam, 0.5, 1.0))

    def restore_decay(self) -> None:
        self.decay_lambda = self.base_decay

    # -------------------------------------------------------------- posterior
    @property
    def alpha(self) -> np.ndarray:
        return np.maximum(self.prior_alpha + self.hits, _EPS)

    @property
    def beta(self) -> np.ndarray:
        return np.maximum(self.prior_beta + self.misses, _EPS)

    @property
    def mean(self) -> np.ndarray:
        a, b = self.alpha, self.beta
        return a / (a + b)

    @property
    def std(self) -> np.ndarray:
        a, b = self.alpha, self.beta
        total = a + b
        return np.sqrt((a * b) / (total * total * (total + 1.0)))

    @property
    def evidence(self) -> np.ndarray:
        """Total decayed observation weight per band."""
        return self.hits + self.misses

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One Thompson draw per band."""
        return rng.beta(self.alpha, self.beta)

    # ------------------------------------------------------------- inspection
    def band_state(self, band_id: int) -> Dict[str, float]:
        a = float(self.alpha[band_id])
        b = float(self.beta[band_id])
        total = a + b
        return {
            "alpha": a,
            "beta": b,
            "mean": a / total,
            "std": float(np.sqrt((a * b) / (total * total * (total + 1.0)))),
            "hits": float(self.hits[band_id]),
            "misses": float(self.misses[band_id]),
        }

    def snapshot(self) -> Dict[str, list]:
        return {
            "alpha": self.alpha.round(4).tolist(),
            "beta": self.beta.round(4).tolist(),
            "mean": self.mean.round(4).tolist(),
            "std": self.std.round(4).tolist(),
            "evidence": self.evidence.round(3).tolist(),
        }


__all__ = ["BetaBernoulliLearner"]
