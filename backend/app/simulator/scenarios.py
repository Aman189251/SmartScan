"""Named scenario presets.

A scenario is a reproducible description of a synthetic environment.  Presets
give the operator a short list of meaningful choices in the UI while keeping
every run fully specified by its seed plus these parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .environment import SyntheticEnvironment


@dataclass(slots=True)
class ScenarioPreset:
    key: str
    title: str
    description: str
    scenario: str
    n_bands: int = 32
    n_slots: int = 2000
    base_activity: float = 0.12
    n_active_sources: int = 6
    change_points: List[int] = field(default_factory=list)
    cycle_length: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def build(self, seed: int, **overrides: Any) -> SyntheticEnvironment:
        kwargs: Dict[str, Any] = {
            "n_bands": self.n_bands,
            "n_slots": self.n_slots,
            "scenario": self.scenario,
            "seed": seed,
            "base_activity": self.base_activity,
            "n_active_sources": self.n_active_sources,
            "change_points": list(self.change_points),
            "cycle_length": self.cycle_length,
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return SyntheticEnvironment(**kwargs)


PRESETS: Dict[str, ScenarioPreset] = {
    p.key: p
    for p in [
        ScenarioPreset(
            key="periodic",
            title="Periodic activity",
            description="Regular, repeating activity windows. Tests temporal prediction.",
            scenario="periodic",
            n_active_sources=6,
        ),
        ScenarioPreset(
            key="burst",
            title="Burst activity",
            description="Short, irregular bursts. Tests reaction speed to brief opportunities.",
            scenario="burst",
            n_active_sources=6,
        ),
        ScenarioPreset(
            key="intermittent",
            title="Intermittent activity",
            description="Markov on/off behaviour with no fixed period. Tests handling of uncertainty.",
            scenario="intermittent",
            n_active_sources=6,
        ),
        ScenarioPreset(
            key="agile",
            title="Frequency-agile abstraction",
            description="Activity migrates between abstract bands. Tests adaptation to moving activity.",
            scenario="agile",
            n_active_sources=5,
        ),
        ScenarioPreset(
            key="mixed",
            title="Mixed environment",
            description="Periodic, burst, intermittent and agile sources together. Default demonstration.",
            scenario="mixed",
            n_active_sources=8,
        ),
        ScenarioPreset(
            key="shift",
            title="Environment shift",
            description="Activity distribution is redrawn at a change point. Tests adaptive recovery.",
            scenario="shift",
            n_active_sources=8,
            change_points=[1000],
        ),
        ScenarioPreset(
            key="noise",
            title="Noise-dominant environment",
            description="Sparse activity against a noisy background. Tests false-alarm robustness.",
            scenario="noise",
            base_activity=0.20,
            n_active_sources=4,
        ),
    ]
}

DEFAULT_PRESET = "mixed"


def get_preset(key: str) -> ScenarioPreset:
    if key not in PRESETS:
        raise KeyError(f"unknown scenario preset {key!r}; available: {sorted(PRESETS)}")
    return PRESETS[key]


def list_presets() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in PRESETS.values()]


def build_environment(cfg, seed: int | None = None, preset_key: str | None = None) -> SyntheticEnvironment:
    """Build an environment from a preset plus the ``environment`` config block.

    The preset owns the *shape* of the environment: source mix, density and
    change points.  Configuration owns its *size* and seed.  Shape keys are
    honoured only when the operator sets them explicitly, because forcing one
    set of change points onto every preset would make distinct scenarios
    identical - notably ``mixed`` and ``shift``, whose only difference is that
    ``shift`` redraws its activity pattern part way through the run.
    """
    env_cfg = cfg["environment"]
    key = preset_key or env_cfg.get("scenario", DEFAULT_PRESET)
    preset = PRESETS.get(key, PRESETS[DEFAULT_PRESET])
    overrides: Dict[str, Any] = {
        "n_bands": env_cfg.get("n_bands"),
        "n_slots": env_cfg.get("n_slots"),
        "cycle_length": env_cfg.get("cycle_length"),
    }
    for shape_key in ("base_activity", "n_active_sources", "change_points"):
        if shape_key in env_cfg:
            overrides[shape_key] = env_cfg.get(shape_key)
    return preset.build(
        seed=int(seed if seed is not None else env_cfg.get("seed", 1234)),
        **overrides,
    )


__all__ = ["ScenarioPreset", "PRESETS", "DEFAULT_PRESET", "get_preset", "list_presets", "build_environment"]
