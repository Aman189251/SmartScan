"""Metrics and model endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from ..config import DATASET_DIR
from ..db.database import sync_model_registry
from ..ml.dataset import DATASET_VERSION, build_dataset
from ..ml.predict import load_predictor
from ..ml.train import feature_importance, train_model
from .schemas import TrainRequest
from .state import AppState, Job, get_state

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Full metric summary for the current or most recent run."""
    return state.controller.summary()


@router.get("/metrics/curves")
def curves(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    snapshot = state.controller.snapshot(include_grid=False)
    return {
        "curves": snapshot.get("curves", {}),
        "bands": snapshot.get("bands", {}),
    }


@router.get("/models")
def models(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Registry contents plus the model currently loaded into the engine."""
    engine = state.controller.engine
    loaded = engine.predictor.describe() if engine is not None else load_predictor("xgboost").describe()
    return {
        "loaded": loaded,
        "registry": state.registry.list_models(),
        "pointers": {
            "xgboost": state.registry.latest_id("xgboost"),
            "random_forest": state.registry.latest_id("random_forest"),
        },
    }


@router.get("/models/{model_id}")
def model_detail(model_id: str, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    try:
        model, meta = state.registry.load(model_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"model {model_id} not found") from exc
    detail = meta.to_dict()
    detail["feature_importance"] = feature_importance(model)
    return detail


@router.post("/models/{model_id}/promote")
def promote(model_id: str, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    try:
        state.registry.promote(model_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"promoted": model_id}


@router.post("/models/train")
def train(request: TrainRequest, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Generate a dataset and train the offline models on a worker thread."""
    active = state.jobs.active("training")
    if active is not None:
        raise HTTPException(status_code=409, detail=f"training job {active.id} already running")

    cfg = state.cfg

    def task(job: Job) -> Dict[str, Any]:
        def dataset_progress(done: int, total: int, scenario: str) -> None:
            job.progress = 0.6 * done / max(1, total)
            job.message = f"simulating {scenario} ({done}/{total} runs)"

        job.message = "generating historical simulation dataset"
        bundle = build_dataset(
            cfg,
            runs_per_scenario=request.runs_per_scenario,
            scenarios=request.scenarios,
            base_seed=request.base_seed,
            bands_per_slot=request.bands_per_slot,
            output=DATASET_DIR / f"training_{DATASET_VERSION}.npz",
            progress=dataset_progress,
        )

        job.message = "training XGBoost"
        job.progress = 0.7
        results = {"xgboost": train_model("xgboost", bundle, cfg, state.registry).to_dict()}

        if request.train_random_forest:
            job.message = "training Random Forest baseline"
            job.progress = 0.9
            results["random_forest"] = train_model(
                "random_forest", bundle, cfg, state.registry).to_dict()

        sync_model_registry(cfg, state.registry)
        job.message = "training complete"
        return {
            "dataset": {
                "rows": len(bundle),
                "positive_rate": round(float(bundle.y.mean()), 4),
                "scenarios": bundle.metadata.get("scenarios", []),
                "label_source": bundle.metadata.get("label_source"),
            },
            "models": results,
        }

    job = state.jobs.submit("training", task, detail="dataset generation and offline training")
    return job.to_dict()


@router.get("/jobs")
def jobs(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return {"jobs": state.jobs.list()}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job.to_dict()


__all__ = ["router"]
