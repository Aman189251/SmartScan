"""Simulation lifecycle and live-state endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from ..db.database import recent_runs, run_detail
from ..scheduler import describe_schedulers
from ..simulator.scenarios import list_presets
from .schemas import SpeedRequest, StartRequest
from .state import AppState, get_state

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/start")
def start(request: StartRequest, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    try:
        return state.controller.start(
            strategy=request.strategy,
            scenario=request.scenario,
            seed=request.seed,
            adaptive=request.adaptive,
            speed=request.speed,
            max_slots=request.max_slots,
            persist=request.persist,
            overrides=request.overrides,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pause")
def pause(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.pause()


@router.post("/resume")
def resume(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.resume()


@router.post("/stop")
def stop(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.stop(join=True)


@router.post("/reset")
def reset(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.reset()


@router.post("/speed")
def set_speed(request: SpeedRequest, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.set_speed(request.speed)


@router.get("/state")
def get_simulation_state(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.status()


@router.get("/snapshot")
def snapshot(include_grid: bool = True, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return state.controller.snapshot(include_grid=include_grid)


@router.get("/activity")
def activity(width: int = Query(200, ge=10, le=2000),
             state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Recent slice of the simulated activity grid, for the spectrum/time view."""
    return state.controller.activity_window(width=width)


@router.get("/scenarios")
def scenarios() -> Dict[str, Any]:
    return {"scenarios": list_presets()}


@router.get("/strategies")
def strategies() -> Dict[str, Any]:
    return {"strategies": describe_schedulers()}


@router.get("/runs")
def runs(limit: int = Query(25, ge=1, le=200),
         state: AppState = Depends(get_state)) -> Dict[str, Any]:
    return {"runs": recent_runs(state.cfg, limit=limit)}


@router.get("/runs/{run_id}")
def run_by_id(run_id: str, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    detail = run_detail(state.cfg, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return detail


__all__ = ["router"]
