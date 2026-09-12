"""Scheduler registry.

Every policy in this package is benchmarkable on identical scenarios, which is
what makes the comparison in the Experiment Arena meaningful.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import DecisionContext, Scheduler
from .baselines import (
    GreedyMLScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    ThompsonScheduler,
    UCBScheduler,
)
from .smart_scan import SmartScanScheduler

SCHEDULER_CLASSES = {
    RandomScheduler.name: RandomScheduler,
    RoundRobinScheduler.name: RoundRobinScheduler,
    GreedyMLScheduler.name: GreedyMLScheduler,
    UCBScheduler.name: UCBScheduler,
    ThompsonScheduler.name: ThompsonScheduler,
    SmartScanScheduler.name: SmartScanScheduler,
}

#: Order used for benchmarks and the Experiment Arena, weakest baseline first.
BENCHMARK_ORDER: List[str] = [
    "random",
    "round_robin",
    "greedy_ml",
    "ucb",
    "thompson",
    "smart_scan",
]


def build_scheduler(name: str, cfg=None) -> Scheduler:
    """Instantiate a scheduler by key, wiring in its configured parameters."""
    if name not in SCHEDULER_CLASSES:
        raise KeyError(f"unknown scheduler {name!r}; available: {sorted(SCHEDULER_CLASSES)}")
    cls = SCHEDULER_CLASSES[name]
    if cfg is None:
        return cls()
    if cls is UCBScheduler:
        return cls(c=float(cfg.get_path("decision.ucb_c", 1.4)))
    if cls is SmartScanScheduler:
        return cls(
            candidate_k=int(cfg.get_path("ml.candidate_k", 5)),
            exploration_floor=float(cfg.get_path("decision.exploration_floor", 0.02)),
        )
    return cls()


def describe_schedulers() -> List[Dict[str, Any]]:
    return [
        {
            "key": key,
            "title": SCHEDULER_CLASSES[key].title,
            "uses_ml": SCHEDULER_CLASSES[key].uses_ml,
        }
        for key in BENCHMARK_ORDER
    ]


__all__ = [
    "Scheduler",
    "DecisionContext",
    "RandomScheduler",
    "RoundRobinScheduler",
    "GreedyMLScheduler",
    "UCBScheduler",
    "ThompsonScheduler",
    "SmartScanScheduler",
    "SCHEDULER_CLASSES",
    "BENCHMARK_ORDER",
    "build_scheduler",
    "describe_schedulers",
]
