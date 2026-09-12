"""Versioned model registry.

Every artifact carries the metadata needed to reproduce and audit it: model id,
version, dataset version, feature version, training date, scenario coverage,
hyperparameters, validation metrics and calibration metrics.  The engine loads a
model by pointer, never by file path guessing, so swapping a model never means
editing code.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from ..config import MODEL_DIR


@dataclass(slots=True)
class ModelMetadata:
    model_id: str
    kind: str                       # xgboost | random_forest | ...
    version: str
    feature_version: str
    feature_names: List[str]
    dataset_version: str = "unknown"
    trained_at: str = ""
    horizon: int = 1
    label_source: str = "truth"
    scenario_coverage: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    n_train_rows: int = 0
    n_test_rows: int = 0
    python_version: str = field(default_factory=platform.python_version)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Filesystem-backed registry under ``models/``."""

    def __init__(self, root: Path | str = MODEL_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    # ------------------------------------------------------------------ index
    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"models": [], "pointers": {}}
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {"models": [], "pointers": {}}

    def _write_index(self, index: Dict[str, Any]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)

    # ------------------------------------------------------------------- save
    def save(self, model: Any, metadata: ModelMetadata) -> Path:
        if not metadata.trained_at:
            metadata.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        target = self.root / metadata.model_id
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target / "model.joblib")
        with open(target / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, indent=2)

        index = self._read_index()
        index["models"] = [m for m in index["models"] if m.get("model_id") != metadata.model_id]
        index["models"].append({
            "model_id": metadata.model_id,
            "kind": metadata.kind,
            "version": metadata.version,
            "trained_at": metadata.trained_at,
            "feature_version": metadata.feature_version,
            "validation_metrics": metadata.validation_metrics,
        })
        index.setdefault("pointers", {})[metadata.kind] = metadata.model_id
        self._write_index(index)
        return target

    # ------------------------------------------------------------------- load
    def load(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        target = self.root / model_id
        model = joblib.load(target / "model.joblib")
        with open(target / "metadata.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return model, ModelMetadata(**meta)

    def latest_id(self, kind: str = "xgboost") -> Optional[str]:
        return self._read_index().get("pointers", {}).get(kind)

    def load_latest(self, kind: str = "xgboost") -> Optional[Tuple[Any, ModelMetadata]]:
        model_id = self.latest_id(kind)
        if not model_id or not (self.root / model_id / "model.joblib").exists():
            return None
        try:
            return self.load(model_id)
        except Exception:  # pragma: no cover - corrupt artifact
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        index = self._read_index()
        pointers = index.get("pointers", {})
        rows = sorted(index.get("models", []), key=lambda m: m.get("trained_at", ""), reverse=True)
        for row in rows:
            row["is_latest"] = pointers.get(row.get("kind")) == row.get("model_id")
        return rows

    def promote(self, model_id: str) -> None:
        """Point the production slot for this model kind at ``model_id``."""
        _, meta = self.load(model_id)
        index = self._read_index()
        index.setdefault("pointers", {})[meta.kind] = model_id
        self._write_index(index)


def next_model_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{kind}-{stamp}"


__all__ = ["ModelRegistry", "ModelMetadata", "next_model_id"]
