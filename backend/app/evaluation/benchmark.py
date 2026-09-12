"""Experiment engine.

Runs every scheduler against the identical seeded environment and receiver
conditions, repeats across seeds, and reports mean and spread rather than a
single lucky run.  Because the environment is rebuilt from the seed, each policy
faces exactly the same realisation of activity, which is what makes the
comparison defensible.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import EXPERIMENT_DIR, Config
from ..core.engine import SmartScanEngine
from ..db.database import session_scope
from ..db.models import Experiment
from ..scheduler import BENCHMARK_ORDER, SCHEDULER_CLASSES

logger = logging.getLogger(__name__)

#: Metrics reported side by side in the Experiment Arena.
HEADLINE_METRICS = [
    "average_intercept_rate",
    "intercept_ratio",
    "average_intercept_delay",
    "probability_of_detection",
    "probability_of_false_alarm",
    "detection_rate",
    "scan_efficiency",
    "coverage",
    "cumulative_reward",
    "average_reward",
    "percentage_correct_predictions",
    "average_adaptation_time",
]


def run_single(
    cfg: Config,
    strategy: str,
    scenario: str,
    seed: int,
    adaptive: bool = True,
    persist: bool = True,
    max_slots: Optional[int] = None,
    experiment_id: Optional[int] = None,
) -> Dict[str, Any]:
    """One policy, one seed, one scenario."""
    engine = SmartScanEngine(
        cfg=cfg, strategy=strategy, seed=seed, scenario=scenario,
        adaptive_enabled=adaptive, persist=persist, experiment_id=experiment_id,
    )
    summary = engine.run(max_steps=max_slots)
    summary["strategy"] = strategy
    summary["seed"] = seed
    return summary


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of the headline metrics, per strategy."""
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    for strategy, runs in by_strategy.items():
        stats: Dict[str, float] = {"runs": len(runs)}
        for metric in HEADLINE_METRICS:
            values = [r.get(metric) for r in runs]
            values = [float(v) for v in values
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and v is not None and not np.isnan(float(v))]
            if not values:
                continue
            stats[f"{metric}_mean"] = round(float(np.mean(values)), 4)
            stats[f"{metric}_std"] = round(float(np.std(values)), 4)
        out[strategy] = stats
    return out


def run_benchmark(
    cfg: Config,
    scenario: str = "mixed",
    seeds: Sequence[int] = (101, 202, 303),
    strategies: Optional[Sequence[str]] = None,
    adaptive: bool = True,
    persist: bool = True,
    max_slots: Optional[int] = None,
    name: Optional[str] = None,
    progress: Optional[Callable[[int, int, str, int], None]] = None,
) -> Dict[str, Any]:
    """Compare policies across seeds on one scenario."""
    strategy_list = list(strategies) if strategies else list(BENCHMARK_ORDER)
    unknown = [s for s in strategy_list if s not in SCHEDULER_CLASSES]
    if unknown:
        raise KeyError(f"unknown strategies: {unknown}")

    experiment_name = name or f"{scenario}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    experiment_id: Optional[int] = None
    if persist:
        with session_scope(cfg) as session:
            experiment = Experiment(
                name=experiment_name, scenario_key=scenario, seeds=list(seeds),
                strategies=strategy_list, status="RUNNING",
                config={"adaptive": adaptive, "max_slots": max_slots},
            )
            session.add(experiment)
            session.flush()
            experiment_id = experiment.id

    rows: List[Dict[str, Any]] = []
    total = len(seeds) * len(strategy_list)
    done = 0
    for seed in seeds:
        for strategy in strategy_list:
            summary = run_single(
                cfg, strategy=strategy, scenario=scenario, seed=int(seed),
                adaptive=adaptive, persist=persist, max_slots=max_slots,
                experiment_id=experiment_id,
            )
            rows.append(summary)
            done += 1
            if progress is not None:
                progress(done, total, strategy, int(seed))

    summary_block = aggregate(rows)
    result = {
        "name": experiment_name,
        "experiment_id": experiment_id,
        "scenario": scenario,
        "seeds": [int(s) for s in seeds],
        "strategies": strategy_list,
        "adaptive": adaptive,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aggregate": summary_block,
        "runs": rows,
        "ranking": rank_strategies(summary_block),
        "paired": paired_comparison(
            rows, reference="round_robin" if "round_robin" in strategy_list
            else strategy_list[0]),
    }

    path = Path(EXPERIMENT_DIR) / f"{experiment_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, indent=2)
    result["path"] = str(path)

    if persist and experiment_id is not None:
        with session_scope(cfg) as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is not None:
                experiment.status = "FINISHED"
                experiment.summary = _json_safe(
                    {"aggregate": summary_block, "ranking": result["ranking"]}
                )
                experiment.finished_at = datetime.now(timezone.utc)
    return result


def paired_comparison(rows: Sequence[Dict[str, Any]],
                      metric: str = "average_intercept_rate",
                      reference: str = "round_robin") -> Dict[str, Any]:
    """Compare each policy against a reference *on the same seed*.

    Between-seed variance dominates absolute metrics here: one seed's environment
    simply contains more activity than another's, and that spread swamps the
    difference between policies.  Because every policy runs on the identical
    environment realisation for a given seed, the per-seed ratio is a paired
    measurement, which removes that shared variance entirely.  This is the
    comparison worth reporting; the absolute means are context.
    """
    by_seed: Dict[int, Dict[str, float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        by_seed.setdefault(int(row["seed"]), {})[str(row["strategy"])] = float(value)

    usable = {seed: values for seed, values in by_seed.items()
              if reference in values and values[reference] > 0}
    if not usable:
        return {"metric": metric, "reference": reference, "seeds": 0, "strategies": {}}

    strategies = sorted({s for values in usable.values() for s in values})
    out: Dict[str, Any] = {}
    for strategy in strategies:
        ratios = [values[strategy] / values[reference]
                  for values in usable.values() if strategy in values]
        if not ratios:
            continue
        out[strategy] = {
            "ratio_mean": round(float(np.mean(ratios)), 4),
            "ratio_std": round(float(np.std(ratios)), 4),
            "ratio_min": round(float(np.min(ratios)), 4),
            "ratio_max": round(float(np.max(ratios)), 4),
            "wins": int(sum(1 for r in ratios if r > 1.0)),
            "seeds": len(ratios),
        }
    return {"metric": metric, "reference": reference, "seeds": len(usable), "strategies": out}


def rank_strategies(aggregate_block: Dict[str, Dict[str, float]],
                    metric: str = "average_intercept_rate_mean") -> List[Dict[str, Any]]:
    """Order strategies by a headline metric, best first."""
    rows = [
        {"strategy": strategy, "value": stats.get(metric)}
        for strategy, stats in aggregate_block.items()
        if stats.get(metric) is not None
    ]
    rows.sort(key=lambda r: r["value"], reverse=True)
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {"strategy": "smart_scan", "adaptive": True,
             "label": "Complete Smart Scan"},
    "no_adaptive": {"strategy": "smart_scan", "adaptive": False,
                    "label": "Smart Scan without the adaptive layer"},
    "math_only": {"strategy": "thompson", "adaptive": False,
                  "label": "Mathematical layer only (no ML prior)"},
    "ml_only": {"strategy": "greedy_ml", "adaptive": False,
                "label": "Offline model only (no exploration, no belief)"},
}


def run_ablation(
    cfg: Config,
    scenario: str = "shift",
    seeds: Sequence[int] = (101, 202, 303),
    persist: bool = True,
    max_slots: Optional[int] = None,
    progress: Optional[Callable[[int, int, str, int], None]] = None,
) -> Dict[str, Any]:
    """Isolate the contribution of each layer on identical environments."""
    rows: List[Dict[str, Any]] = []
    total = len(seeds) * len(ABLATIONS)
    done = 0
    for seed in seeds:
        for key, spec in ABLATIONS.items():
            summary = run_single(
                cfg, strategy=spec["strategy"], scenario=scenario, seed=int(seed),
                adaptive=bool(spec["adaptive"]), persist=persist, max_slots=max_slots,
            )
            summary["strategy"] = key            # label the arm, not the policy
            summary["policy"] = spec["strategy"]
            summary["label"] = spec["label"]
            rows.append(summary)
            done += 1
            if progress is not None:
                progress(done, total, key, int(seed))

    block = aggregate(rows)
    result = {
        "name": f"ablation-{scenario}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "scenario": scenario,
        "seeds": [int(s) for s in seeds],
        "arms": {k: v["label"] for k, v in ABLATIONS.items()},
        "aggregate": block,
        "ranking": rank_strategies(block),
        "paired": paired_comparison(rows, reference="math_only"),
        "runs": rows,
    }
    path = Path(EXPERIMENT_DIR) / f"{result['name']}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, indent=2)
    result["path"] = str(path)
    return result


def _json_safe(value: Any) -> Any:
    import math

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


__all__ = ["run_benchmark", "run_single", "run_ablation", "aggregate",
           "rank_strategies", "paired_comparison", "HEADLINE_METRICS", "ABLATIONS"]
