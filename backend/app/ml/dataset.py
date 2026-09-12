"""Historical dataset generation for the offline model.

Training data comes from seeded synthetic runs driven by an exploration-heavy
policy, so the feature space is covered rather than concentrated on whatever a
greedy policy happens to like.  Features are built exclusively from the
observation history available *before* the prediction target.  Ground truth
enters only as the label, and only here, in the offline path.

Two labelling modes are supported:

``truth``
    y = hidden activity at ``t + horizon``.  This is the offline-synthetic
    formulation from the design document.

``observation``
    y = whether the next real scan of that band at or after ``t + horizon``
    reported a detection.  Nothing but observations is used, which matches the
    problem statement's "trained based on hits and misses" and is the mode to
    use if a dataset ever has to be built from recorded receiver data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DATASET_DIR, Config
from ..core.types import ScanCommand
from ..simulator.receiver import ReceiverSimulator
from ..simulator.scenarios import PRESETS, get_preset
from .features import FEATURE_NAMES, FEATURE_VERSION, FeatureEngine

DATASET_VERSION = "ds-1.0"


@dataclass(slots=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray            # run index, used for leakage-free splitting
    slots: np.ndarray
    bands: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X, y=self.y, groups=self.groups, slots=self.slots, bands=self.bands,
            feature_names=np.array(FEATURE_NAMES, dtype=object),
        )
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DatasetBundle":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        meta_path = path.with_suffix(".meta.json")
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        return cls(
            X=data["X"], y=data["y"], groups=data["groups"],
            slots=data["slots"], bands=data["bands"], metadata=metadata,
        )

    def split_by_group(self, val_fraction: float = 0.2, test_fraction: float = 0.2,
                       seed: int = 0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Group-wise split so no run contributes to two partitions."""
        unique = np.unique(self.groups)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        n = len(unique)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        test_g = set(unique[:n_test].tolist())
        val_g = set(unique[n_test:n_test + n_val].tolist())

        test_mask = np.isin(self.groups, list(test_g))
        val_mask = np.isin(self.groups, list(val_g))
        train_mask = ~(test_mask | val_mask)
        return {
            "train": (self.X[train_mask], self.y[train_mask]),
            "val": (self.X[val_mask], self.y[val_mask]),
            "test": (self.X[test_mask], self.y[test_mask]),
        }


class ExplorationPolicy:
    """Coverage-oriented behaviour policy used only for dataset collection."""

    def __init__(self, n_bands: int, rng: np.random.Generator):
        self.n_bands = n_bands
        self.rng = rng
        self._cursor = 0

    def select(self, available: Sequence[int], engine: FeatureEngine) -> int:
        mode = self.rng.random()
        if mode < 0.45:                                  # systematic sweep
            for _ in range(self.n_bands):
                band = self._cursor % self.n_bands
                self._cursor += 1
                if band in available:
                    return band
        if mode < 0.75:                                  # uniform random
            return int(self.rng.choice(list(available)))
        # Mild exploitation so the dataset also contains high-activity regions.
        rates = (engine.detections + 0.5) / (engine.scans + 1.0)
        avail = np.asarray(list(available), dtype=int)
        noisy = rates[avail] + self.rng.normal(0.0, 0.12, size=len(avail))
        return int(avail[int(np.argmax(noisy))])


def generate_run(
    preset_key: str,
    seed: int,
    cfg: Config,
    run_index: int,
    bands_per_slot: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one run and emit (features, labels, slots, bands)."""
    preset = get_preset(preset_key)
    env = preset.build(seed=seed)
    receiver = ReceiverSimulator(
        env,
        pd=float(cfg.get_path("receiver.pd", 0.9)),
        pfa=float(cfg.get_path("receiver.pfa", 0.03)),
        capacity=int(cfg.get_path("receiver.capacity", 1)),
        scan_duration=int(cfg.get_path("receiver.scan_duration", 1)),
        revisit_lockout=int(cfg.get_path("receiver.revisit_lockout", 0)),
        seed=seed + 991,
    )
    engine = FeatureEngine(
        n_bands=env.n_bands,
        short_window=int(cfg.get_path("features.short_window", 8)),
        medium_window=int(cfg.get_path("features.medium_window", 24)),
        long_window=int(cfg.get_path("features.long_window", 64)),
        ewma_fast=float(cfg.get_path("features.ewma_fast", 0.30)),
        ewma_slow=float(cfg.get_path("features.ewma_slow", 0.05)),
        cycle_length=int(cfg.get_path("features.cycle_length", 120)),
    )
    horizon = int(cfg.get_path("ml.horizon", 1))
    label_source = str(cfg.get_path("ml.label_source", "truth"))
    rng = np.random.default_rng(seed + 5077)
    policy = ExplorationPolicy(env.n_bands, rng)

    rows_x: List[np.ndarray] = []
    rows_slot: List[int] = []
    rows_band: List[int] = []
    scan_log: List[Tuple[int, int, int]] = []   # (slot, band, detected)

    for slot in range(env.n_slots - horizon):
        available = receiver.available_bands(slot)
        if not available:
            continue
        matrix = engine.build_matrix(slot)
        band = policy.select(available, engine)

        sample = {band}
        if bands_per_slot > 1:
            extra = rng.choice(env.n_bands, size=min(bands_per_slot - 1, env.n_bands),
                               replace=False)
            sample.update(int(b) for b in extra)
        for b in sample:
            rows_x.append(matrix[b])
            rows_slot.append(slot)
            rows_band.append(int(b))

        observation = receiver.observe(ScanCommand(run_id=f"ds{run_index}", slot=slot, band_id=band))
        engine.observe(slot, band, observation.detected)
        scan_log.append((slot, band, int(observation.detected)))

    X = np.asarray(rows_x, dtype=np.float32)
    slots = np.asarray(rows_slot, dtype=np.int32)
    bands = np.asarray(rows_band, dtype=np.int16)

    if label_source == "truth":
        y = env.truth[slots + horizon, bands].astype(np.uint8)
        keep = np.ones(len(y), dtype=bool)
    else:
        y, keep = _observation_labels(scan_log, slots, bands, horizon, env.n_bands)

    return X[keep], y[keep], slots[keep], bands[keep]


def _observation_labels(scan_log: List[Tuple[int, int, int]], slots: np.ndarray,
                        bands: np.ndarray, horizon: int,
                        n_bands: int) -> Tuple[np.ndarray, np.ndarray]:
    """Label each row by the next observed outcome for that band."""
    per_band: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(n_bands)}
    for slot, band, detected in scan_log:
        per_band[band].append((slot, detected))

    y = np.zeros(len(slots), dtype=np.uint8)
    keep = np.zeros(len(slots), dtype=bool)
    for i, (slot, band) in enumerate(zip(slots, bands)):
        entries = per_band[int(band)]
        target = int(slot) + horizon
        lo, hi = 0, len(entries)
        while lo < hi:                                  # first scan at or after target
            mid = (lo + hi) // 2
            if entries[mid][0] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(entries):
            y[i] = entries[lo][1]
            keep[i] = True
    return y, keep


def build_dataset(
    cfg: Config,
    runs_per_scenario: int = 3,
    scenarios: Optional[Sequence[str]] = None,
    base_seed: int = 4242,
    bands_per_slot: int = 6,
    output: Optional[Path | str] = None,
    progress=None,
) -> DatasetBundle:
    """Generate a training dataset spanning several scenarios and seeds."""
    scenario_keys = list(scenarios) if scenarios else list(PRESETS.keys())
    xs, ys, gs, ss, bs = [], [], [], [], []
    run_index = 0
    total = len(scenario_keys) * runs_per_scenario

    for key in scenario_keys:
        for r in range(runs_per_scenario):
            seed = base_seed + 1000 * run_index + r
            X, y, slots, bands = generate_run(key, seed, cfg, run_index, bands_per_slot)
            xs.append(X)
            ys.append(y)
            gs.append(np.full(len(y), run_index, dtype=np.int32))
            ss.append(slots)
            bs.append(bands)
            run_index += 1
            if progress is not None:
                progress(run_index, total, key)

    bundle = DatasetBundle(
        X=np.concatenate(xs), y=np.concatenate(ys), groups=np.concatenate(gs),
        slots=np.concatenate(ss), bands=np.concatenate(bs),
        metadata={
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scenarios": scenario_keys,
            "runs_per_scenario": runs_per_scenario,
            "base_seed": base_seed,
            "horizon": int(cfg.get_path("ml.horizon", 1)),
            "label_source": str(cfg.get_path("ml.label_source", "truth")),
            "receiver": {
                "pd": float(cfg.get_path("receiver.pd", 0.9)),
                "pfa": float(cfg.get_path("receiver.pfa", 0.03)),
            },
        },
    )
    bundle.metadata["n_rows"] = len(bundle)
    bundle.metadata["positive_rate"] = float(bundle.y.mean())

    path = Path(output) if output else DATASET_DIR / f"training_{DATASET_VERSION}.npz"
    bundle.save(path)
    bundle.metadata["path"] = str(path)
    return bundle


__all__ = ["build_dataset", "generate_run", "DatasetBundle", "DATASET_VERSION"]
