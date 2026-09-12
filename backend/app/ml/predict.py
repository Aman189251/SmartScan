"""Prediction service (Layer 1 at run time).

The model is loaded once at application start and reused for every decision, as
required by the performance plan.  Prediction is a single batched call over all
bands, not one call per band.

If no trained artifact exists yet the engine still runs: a transparent heuristic
predictor stands in, and both the API and the UI report that no model is loaded
so a demonstration can never be mistaken for a trained result.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .features import FEATURE_NAMES, FEATURE_VERSION
from .model_registry import ModelMetadata, ModelRegistry


class BasePredictor:
    kind = "base"
    is_trained = False

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "trained": self.is_trained}


class HeuristicPredictor(BasePredictor):
    """Observation-only fallback used when no trained model is available.

    It blends the fast and slow observed rates with the same-phase rate and
    pulls unscanned bands towards an optimistic value so they still get looked
    at.  It is deliberately simple and is never presented as the ML layer.
    """

    kind = "heuristic"
    is_trained = False

    _IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=float)
        ewma_fast = f[:, self._IDX["ewma_fast"]]
        rate_short = f[:, self._IDX["rate_short"]]
        phase_rate = f[:, self._IDX["phase_rate"]]
        never = f[:, self._IDX["never_scanned"]]
        stale = f[:, self._IDX["since_scan_norm"]]

        p = 0.40 * ewma_fast + 0.30 * rate_short + 0.30 * phase_rate
        p = p + 0.15 * stale                      # staleness raises the value of a look
        p = np.where(never > 0.5, 0.60, p)        # optimism under complete ignorance
        return np.clip(p, 0.01, 0.99)

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "trained": False,
            "note": "No trained model loaded; using an observation-only heuristic prior.",
        }


class ActivityPredictor(BasePredictor):
    """Wraps a trained scikit-learn compatible classifier from the registry."""

    is_trained = True

    def __init__(self, model: Any, metadata: ModelMetadata):
        self.model = model
        self.metadata = metadata
        self.kind = metadata.kind
        if metadata.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"model {metadata.model_id} was trained on features {metadata.feature_version}, "
                f"engine provides {FEATURE_VERSION}; retrain before use"
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        proba = self.model.predict_proba(x)[:, 1]
        return np.clip(proba.astype(float), 1e-4, 1 - 1e-4)

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "trained": True,
            "model_id": self.metadata.model_id,
            "version": self.metadata.version,
            "trained_at": self.metadata.trained_at,
            "feature_version": self.metadata.feature_version,
            "dataset_version": self.metadata.dataset_version,
            "horizon": self.metadata.horizon,
            "validation_metrics": self.metadata.validation_metrics,
            "calibration_metrics": self.metadata.calibration_metrics,
        }


def load_predictor(kind: str = "xgboost", registry: Optional[ModelRegistry] = None) -> BasePredictor:
    """Load the production model for ``kind``, falling back to the heuristic."""
    registry = registry or ModelRegistry()
    loaded = registry.load_latest(kind)
    if loaded is None:
        return HeuristicPredictor()
    model, metadata = loaded
    try:
        return ActivityPredictor(model, metadata)
    except ValueError:
        return HeuristicPredictor()


__all__ = ["BasePredictor", "ActivityPredictor", "HeuristicPredictor", "load_predictor"]
