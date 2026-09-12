"""Configuration loading for Smart Scan Strategy.

Configuration is a plain nested dictionary loaded from YAML and wrapped in a small
attribute-access helper.  Every subsystem receives the sub-tree it needs, so no
module reaches for global state.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATASET_DIR, MODEL_DIR, EXPERIMENT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Config(Mapping):
    """Read-only mapping with attribute access and dotted lookup."""

    __slots__ = ("_data",)

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"

    # -- Helpers ----------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, overrides: Dict[str, Any] | None) -> "Config":
        """Return a new Config with ``overrides`` deep-merged on top."""
        if not overrides:
            return Config(copy.deepcopy(self._data))
        return Config(_deep_merge(copy.deepcopy(self._data), overrides))


def _deep_merge(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value) if not isinstance(value, Config) else value.to_dict()
    return base


def load_config(path: str | os.PathLike | None = None,
                overrides: Dict[str, Any] | None = None) -> Config:
    """Load configuration from YAML, applying optional overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data).merged(overrides)


def database_url(cfg: Config) -> str:
    """Resolve a relative SQLite URL against the project root."""
    url = cfg.get_path("database.url", "sqlite:///smartscan.db")
    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw = url[len(prefix):]
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / raw
        return prefix + str(candidate)
    return url


__all__ = [
    "Config",
    "load_config",
    "database_url",
    "PROJECT_ROOT",
    "DATASET_DIR",
    "MODEL_DIR",
    "EXPERIMENT_DIR",
    "LOG_DIR",
    "DEFAULT_CONFIG_PATH",
]
