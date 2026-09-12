"""Experiment Arena endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import EXPERIMENT_DIR
from ..db.database import list_experiments
from ..evaluation.benchmark import HEADLINE_METRICS, run_ablation, run_benchmark
from .schemas import ExperimentRequest
from .state import AppState, Job, get_state

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/run")
def run_experiment(request: ExperimentRequest,
                   state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Run every selected policy against identical seeded environments."""
    active = state.jobs.active("experiment")
    if active is not None:
        raise HTTPException(status_code=409, detail=f"experiment {active.id} already running")

    cfg = state.cfg

    def task(job: Job) -> Dict[str, Any]:
        def progress(done: int, total: int, strategy: str, seed: int) -> None:
            job.progress = done / max(1, total)
            job.message = f"{strategy} on seed {seed} ({done}/{total})"

        if request.ablation:
            return run_ablation(
                cfg, scenario=request.scenario, seeds=request.seeds,
                max_slots=request.max_slots, progress=progress,
            )
        return run_benchmark(
            cfg, scenario=request.scenario, seeds=request.seeds,
            strategies=request.strategies, adaptive=request.adaptive,
            max_slots=request.max_slots, name=request.name, progress=progress,
        )

    kind = "ablation" if request.ablation else "benchmark"
    job = state.jobs.submit("experiment", task, detail=f"{kind} on {request.scenario}")
    return job.to_dict()


@router.get("")
def experiments(limit: int = Query(25, ge=1, le=100),
                state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return {
        "experiments": list_experiments(state.cfg, limit=limit),
        "headline_metrics": HEADLINE_METRICS,
    }


@router.get("/files")
def experiment_files(limit: int = Query(25, ge=1, le=100)) -> Dict[str, Any]:
    """Saved experiment result files, newest first."""
    files = sorted(Path(EXPERIMENT_DIR).glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return {"files": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in files]}


@router.get("/file/{name}")
def experiment_file(name: str) -> Dict[str, Any]:
    path = Path(EXPERIMENT_DIR) / name
    if not path.exists() or path.suffix != ".json" or path.parent != Path(EXPERIMENT_DIR):
        raise HTTPException(status_code=404, detail=f"experiment file {name} not found")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["router"]
