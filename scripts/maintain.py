"""Housekeeping for the database and the model registry.

A benchmark writes one observation, decision, prediction and belief row per slot,
plus a periodic all-band snapshot, so a few dozen runs reach a few hundred
megabytes. Nothing is lost by pruning per-slot detail from old runs: run
summaries, metrics and adaptive events are kept, and those are what the run
history view and the experiment comparisons read.

    python scripts/maintain.py --status
    python scripts/maintain.py --prune-details --keep 10
    python scripts/maintain.py --prune-models --keep 2
    python scripts/maintain.py --vacuum
    python scripts/maintain.py --reset          # destructive, asks first
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from sqlalchemy import delete, func, select, text

from backend.app.config import MODEL_DIR, PROJECT_ROOT, database_url, load_config
from backend.app.db.database import get_engine, session_scope
from backend.app.db.models import (
    Belief,
    DecisionRecord,
    Observation,
    Prediction,
    Run,
)
from backend.app.ml.model_registry import ModelRegistry

DETAIL_TABLES = (Observation, DecisionRecord, Prediction, Belief)


def database_path(cfg) -> Path:
    url = database_url(cfg)
    return Path(url.replace("sqlite:///", "")) if url.startswith("sqlite:///") else Path()


def status(cfg) -> None:
    path = database_path(cfg)
    size = path.stat().st_size / 1024 / 1024 if path.exists() else 0.0
    print(f"database: {path}  ({size:.1f} MB)")
    with session_scope(cfg) as session:
        runs = session.scalar(select(func.count()).select_from(Run)) or 0
        print(f"  runs: {runs}")
        for table in DETAIL_TABLES:
            count = session.scalar(select(func.count()).select_from(table)) or 0
            print(f"  {table.__tablename__}: {count:,} rows")

    models = list(Path(MODEL_DIR).glob("*/model.joblib"))
    total = sum(m.stat().st_size for m in models) / 1024 / 1024
    print(f"model registry: {len(models)} artifacts ({total:.1f} MB)")
    registry = ModelRegistry()
    for kind in ("xgboost", "random_forest"):
        print(f"  production {kind}: {registry.latest_id(kind)}")


def prune_details(cfg, keep: int) -> None:
    """Drop per-slot rows for all but the newest ``keep`` runs."""
    with session_scope(cfg) as session:
        recent = [r for (r,) in session.execute(
            select(Run.id).order_by(Run.started_at.desc()).limit(keep))]
        stale = [r for (r,) in session.execute(select(Run.id)) if r not in set(recent)]
        if not stale:
            print("nothing to prune")
            return
        removed = 0
        for table in DETAIL_TABLES:
            result = session.execute(delete(table).where(table.run_id.in_(stale)))
            removed += result.rowcount or 0
        print(f"pruned {removed:,} detail rows from {len(stale)} run(s); "
              f"kept summaries, metrics and adaptive events")


def prune_models(cfg, keep: int) -> None:
    """Keep the newest ``keep`` artifacts per kind, never the production pointer."""
    registry = ModelRegistry()
    rows = registry.list_models()
    protected = {registry.latest_id(k) for k in ("xgboost", "random_forest")}
    by_kind: dict[str, list] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)

    removed = 0
    for kind, entries in by_kind.items():
        entries.sort(key=lambda r: r.get("trained_at", ""), reverse=True)
        for row in entries[keep:]:
            model_id = row["model_id"]
            if model_id in protected:
                continue
            target = Path(MODEL_DIR) / model_id
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
                print(f"  removed {model_id} ({kind})")
    print(f"removed {removed} model artifact(s); the registry index still lists them")


def vacuum(cfg) -> None:
    path = database_path(cfg)
    before = path.stat().st_size / 1024 / 1024 if path.exists() else 0.0
    engine = get_engine(cfg)
    with engine.connect() as connection:
        connection.execute(text("VACUUM"))
    after = path.stat().st_size / 1024 / 1024 if path.exists() else 0.0
    print(f"vacuum: {before:.1f} MB -> {after:.1f} MB")


def reset(cfg, assume_yes: bool) -> None:
    path = database_path(cfg)
    if not path.exists():
        print("no database to remove")
        return
    if not assume_yes:
        answer = input(f"Delete {path} and all run history? [y/N] ").strip().lower()
        if answer != "y":
            print("cancelled")
            return
    get_engine(cfg).dispose()
    path.unlink()
    print(f"removed {path}; it will be recreated on the next run")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Scan housekeeping")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--prune-details", action="store_true")
    parser.add_argument("--prune-models", action="store_true")
    parser.add_argument("--keep", type=int, default=10,
                        help="runs (or models per kind) to keep")
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the reset confirmation")
    args = parser.parse_args()

    cfg = load_config()
    if args.reset:
        reset(cfg, args.yes)
        return 0
    if args.prune_details:
        prune_details(cfg, args.keep)
    if args.prune_models:
        prune_models(cfg, max(1, args.keep))
    if args.vacuum:
        vacuum(cfg)
    if args.status or not any((args.prune_details, args.prune_models, args.vacuum)):
        status(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
