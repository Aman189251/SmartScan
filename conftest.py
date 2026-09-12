"""Pytest configuration: project root on sys.path plus shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Small, fast configuration writing to a throwaway database."""
    return load_config(overrides={
        "environment": {"n_bands": 12, "n_slots": 300, "n_active_sources": 4,
                        "change_points": [150], "seed": 4321},
        "database": {"url": f"sqlite:///{tmp_path.joinpath('test.db').as_posix()}"},
        "adaptive": {"window": 40, "acceptance": {"min_observations": 40}},
    })


@pytest.fixture
def small_env(cfg):
    from backend.app.simulator.scenarios import build_environment

    return build_environment(cfg, seed=99, preset_key="mixed")
