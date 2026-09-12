"""Scheduler, prediction, belief and adaptive-layer endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from ..core.types import band_label
from .state import AppState, get_state

router = APIRouter(tags=["scheduler"])


@router.get("/scheduler/state")
def scheduler_state(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Current decision plus the explanation the operator sees."""
    status = state.controller.status()
    return {
        "state": status.get("state"),
        "strategy": status.get("strategy"),
        "slot": status.get("slot"),
        "current": status.get("current"),
        "receiver": status.get("receiver"),
    }


@router.get("/predictions")
def predictions(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Latest offline-model probability for every band."""
    snapshot = state.controller.snapshot(include_grid=True)
    values: List[float] = snapshot.get("ml_probabilities", []) or []
    ranked = sorted(range(len(values)), key=lambda b: values[b], reverse=True)
    return {
        "slot": snapshot.get("slot"),
        "model": snapshot.get("model"),
        "probabilities": values,
        "top": [
            {"band": b, "label": band_label(b), "probability": values[b]}
            for b in ranked[:8]
        ],
    }


@router.get("/beliefs")
def beliefs(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Current Beta posterior mean and uncertainty for every band."""
    snapshot = state.controller.snapshot(include_grid=True)
    means: List[float] = snapshot.get("posterior_means", []) or []
    stds: List[float] = snapshot.get("posterior_stds", []) or []
    return {
        "slot": snapshot.get("slot"),
        "posterior_means": means,
        "posterior_stds": stds,
        "bands": [
            {"band": b, "label": band_label(b), "mean": means[b],
             "std": stds[b] if b < len(stds) else None}
            for b in range(len(means))
        ],
    }


@router.get("/scans")
def scans(limit: int = Query(120, ge=1, le=400),
          state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Recent scan timeline."""
    snapshot = state.controller.snapshot(include_grid=False)
    timeline = snapshot.get("timeline", []) or []
    return {"scans": timeline[-limit:], "count": len(timeline)}


@router.get("/adaptive")
def adaptive(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Hidden adaptive layer: state, monitors, drift and acceptance criteria."""
    return state.controller.adaptive_status()


__all__ = ["router"]
