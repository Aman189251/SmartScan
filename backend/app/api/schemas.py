"""Pydantic request/response models for the local service layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    strategy: str = Field("smart_scan", description="Scheduling policy key")
    scenario: str = Field("mixed", description="Scenario preset key")
    seed: Optional[int] = Field(None, description="Run seed; omit to use the configured seed")
    adaptive: bool = Field(True, description="Enable the hidden adaptive layer")
    speed: float = Field(0.0, ge=0.0, le=10_000.0,
                         description="Slots per second; 0 runs as fast as possible")
    max_slots: Optional[int] = Field(None, ge=1, description="Stop after this many slots")
    persist: bool = Field(True, description="Write this run to the database")
    overrides: Optional[Dict[str, Any]] = Field(
        None, description="Deep-merged configuration overrides for this run only")


class SpeedRequest(BaseModel):
    speed: float = Field(0.0, ge=0.0, le=10_000.0)


class ExperimentRequest(BaseModel):
    scenario: str = "mixed"
    seeds: List[int] = Field(default_factory=lambda: [101, 202, 303])
    strategies: Optional[List[str]] = None
    adaptive: bool = True
    max_slots: Optional[int] = Field(None, ge=1)
    name: Optional[str] = None
    ablation: bool = Field(False, description="Run the layer-ablation arms instead")


class TrainRequest(BaseModel):
    runs_per_scenario: int = Field(2, ge=1, le=20)
    scenarios: Optional[List[str]] = None
    base_seed: int = 4242
    bands_per_slot: int = Field(6, ge=1, le=32)
    train_random_forest: bool = True


class StatusResponse(BaseModel):
    state: str
    run_id: Optional[str] = None
    slot: int = 0
    model_config = {"extra": "allow"}


__all__ = ["StartRequest", "SpeedRequest", "ExperimentRequest", "TrainRequest", "StatusResponse"]
