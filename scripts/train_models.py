"""Generate the historical dataset and train the offline models.

    python scripts/train_models.py --runs-per-scenario 3

Writes the dataset to ``datasets/`` and registers both models under ``models/``,
promoting each to the production pointer for its kind.
"""

from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401  (sys.path side effect)

from backend.app.config import DATASET_DIR, load_config
from backend.app.db.database import init_database, sync_model_registry
from backend.app.ml.dataset import DATASET_VERSION, DatasetBundle, build_dataset
from backend.app.ml.model_registry import ModelRegistry
from backend.app.ml.train import feature_importance, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Smart Scan offline models")
    parser.add_argument("--runs-per-scenario", type=int, default=3)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--bands-per-slot", type=int, default=6)
    parser.add_argument("--slots", type=int, default=None,
                        help="Override slots per training run")
    parser.add_argument("--reuse-dataset", action="store_true",
                        help="Load the existing dataset instead of regenerating it")
    parser.add_argument("--skip-random-forest", action="store_true")
    args = parser.parse_args()

    overrides = {"environment": {"n_slots": args.slots}} if args.slots else None
    cfg = load_config(overrides=overrides)
    init_database(cfg)
    registry = ModelRegistry()
    dataset_path = DATASET_DIR / f"training_{DATASET_VERSION}.npz"

    started = time.perf_counter()
    if args.reuse_dataset and dataset_path.exists():
        print(f"[dataset] loading {dataset_path}")
        bundle = DatasetBundle.load(dataset_path)
    else:
        def progress(done: int, total: int, scenario: str) -> None:
            print(f"[dataset] {done}/{total} runs simulated (scenario: {scenario})", flush=True)

        bundle = build_dataset(
            cfg,
            runs_per_scenario=args.runs_per_scenario,
            scenarios=args.scenarios,
            base_seed=args.base_seed,
            bands_per_slot=args.bands_per_slot,
            output=dataset_path,
            progress=progress,
        )
    print(f"[dataset] {len(bundle):,} rows, positive rate "
          f"{float(bundle.y.mean()):.3f}, {time.perf_counter() - started:.1f}s")

    print("[train] XGBoost")
    xgb_result = train_model("xgboost", bundle, cfg, registry)
    print(json.dumps(xgb_result.metrics, indent=2))
    model, _ = registry.load(xgb_result.model_id)
    print("[train] top features:", json.dumps(feature_importance(model, 8), indent=2))

    if not args.skip_random_forest:
        print("[train] Random Forest baseline")
        rf_result = train_model("random_forest", bundle, cfg, registry)
        print(json.dumps(rf_result.metrics, indent=2))

    added = sync_model_registry(cfg, registry)
    print(f"[registry] mirrored {added} new model(s) into the database")
    print(f"[done] total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
