"""Run a headless benchmark or ablation and print the comparison table.

    python scripts/run_experiment.py --scenario shift --seeds 101 202 303
    python scripts/run_experiment.py --ablation --scenario shift
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from backend.app.config import load_config
from backend.app.db.database import init_database
from backend.app.evaluation.benchmark import run_ablation, run_benchmark
from backend.app.scheduler import BENCHMARK_ORDER

COLUMNS = [
    ("average_intercept_rate", "intercept/slot", 4),
    ("intercept_ratio", "intercepted", 4),
    ("average_intercept_delay", "delay", 2),
    ("probability_of_detection", "Pd", 3),
    ("probability_of_false_alarm", "Pfa", 3),
    ("scan_efficiency", "efficiency", 3),
    ("coverage", "coverage", 3),
    ("cumulative_reward", "reward", 1),
]


def print_table(result: dict) -> None:
    aggregate = result.get("aggregate", {})
    arms = result.get("arms", {})
    header = f"{'strategy':<24}" + "".join(f"{label:>16}" for _, label, _ in COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    rows = sorted(aggregate.items(),
                  key=lambda kv: kv[1].get("average_intercept_rate_mean", 0.0), reverse=True)
    for strategy, stats in rows:
        name = arms.get(strategy, strategy)[:23]
        line = f"{name:<24}"
        for key, _, digits in COLUMNS:
            mean = stats.get(f"{key}_mean")
            line += f"{'--':>16}" if mean is None else f"{mean:>16.{digits}f}"
        print(line)
    print()
    print_paired(result)


def print_paired(result: dict) -> None:
    """Per-seed comparison against a reference policy.

    Absolute metrics carry large between-seed variance because one seed's
    environment holds more activity than another's. Every policy sees the
    identical environment for a given seed, so the per-seed ratio removes that
    shared variance and is the comparison to read.
    """
    paired = result.get("paired") or {}
    strategies = paired.get("strategies") or {}
    if not strategies:
        return
    arms = result.get("arms", {})
    reference = paired.get("reference")
    print(f"Paired per-seed comparison on {paired.get('metric')} "
          f"against '{reference}' over {paired.get('seeds')} seeds")
    print(f"{'strategy':<24}{'ratio':>10}{'std':>10}{'min':>10}{'max':>10}{'wins':>8}")
    print("-" * 72)
    for strategy, stats in sorted(strategies.items(),
                                  key=lambda kv: kv[1]["ratio_mean"], reverse=True):
        name = arms.get(strategy, strategy)[:23]
        print(f"{name:<24}{stats['ratio_mean']:>10.2f}{stats['ratio_std']:>10.2f}"
              f"{stats['ratio_min']:>10.2f}{stats['ratio_max']:>10.2f}"
              f"{stats['wins']:>5}/{stats['seeds']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Smart Scan against baselines")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--strategies", nargs="*", default=None,
                        help=f"subset of {BENCHMARK_ORDER}")
    parser.add_argument("--slots", type=int, default=1200)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    init_database(cfg)

    def progress(done: int, total: int, strategy: str, seed: int) -> None:
        print(f"  [{done}/{total}] {strategy} seed={seed}", flush=True)

    print(f"Scenario: {args.scenario}  seeds: {args.seeds}  slots: {args.slots}")
    if args.ablation:
        result = run_ablation(cfg, scenario=args.scenario, seeds=args.seeds,
                              persist=not args.no_persist, max_slots=args.slots,
                              progress=progress)
    else:
        result = run_benchmark(cfg, scenario=args.scenario, seeds=args.seeds,
                               strategies=args.strategies, adaptive=not args.no_adaptive,
                               persist=not args.no_persist, max_slots=args.slots,
                               progress=progress)
    print_table(result)
    print(f"Saved to {result.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
