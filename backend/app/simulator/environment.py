"""Synthetic multi-band environment with hidden ground truth.

The environment owns *reality*: for every abstract band and every discrete time
slot it holds a hidden binary activity state.  Only the receiver simulator and
the evaluation engine are allowed to read it.  Schedulers, the feature engine,
the ML model and the Bayesian learner never receive this object.

Bands are abstract sensing regions (B01..BNN).  No real operating frequency,
emitter identity or equipment parameter is modelled anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCENARIOS = ("periodic", "burst", "intermittent", "agile", "mixed", "shift", "noise")


@dataclass(slots=True)
class ActivitySource:
    """One synthetic activity source occupying one or more abstract bands."""

    source_id: int
    kind: str                      # periodic | burst | intermittent | agile
    bands: List[int]
    period: int = 60
    duty: int = 12
    phase: int = 0
    p_on: float = 0.05             # inactive -> active transition probability
    p_off: float = 0.25            # active -> inactive transition probability
    hop_interval: int = 40         # slots between band hops (agile sources)
    intensity: float = 1.0

    def describe(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "bands": list(self.bands),
            "period": self.period,
            "duty": self.duty,
            "phase": self.phase,
            "hop_interval": self.hop_interval,
        }


@dataclass(slots=True)
class EnvironmentSegment:
    """A stretch of slots governed by one draw of source parameters."""

    start: int
    end: int
    sources: List[ActivitySource] = field(default_factory=list)


class SyntheticEnvironment:
    """Generates and holds the hidden activity grid for one seeded run."""

    def __init__(
        self,
        n_bands: int = 32,
        n_slots: int = 2000,
        scenario: str = "mixed",
        seed: int = 1234,
        base_activity: float = 0.12,
        n_active_sources: int = 6,
        change_points: Sequence[int] | None = None,
        cycle_length: int = 120,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        self.n_bands = int(n_bands)
        self.n_slots = int(n_slots)
        self.scenario = scenario
        self.seed = int(seed)
        self.base_activity = float(base_activity)
        self.n_active_sources = int(n_active_sources)
        self.cycle_length = int(cycle_length)
        self.change_points: List[int] = sorted(
            {int(cp) for cp in (change_points or []) if 0 < int(cp) < self.n_slots}
        )
        if scenario == "shift" and not self.change_points:
            self.change_points = [self.n_slots // 2]

        self._rng = np.random.default_rng(self.seed)
        self.segments: List[EnvironmentSegment] = []
        self.truth: np.ndarray = np.zeros((self.n_slots, self.n_bands), dtype=np.uint8)
        self._generate()

    # ------------------------------------------------------------------ build
    def _segment_bounds(self) -> List[Tuple[int, int]]:
        edges = [0, *self.change_points, self.n_slots]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def _draw_sources(self) -> List[ActivitySource]:
        rng = self._rng
        kinds_for_scenario = {
            "periodic": ["periodic"],
            "burst": ["burst"],
            "intermittent": ["intermittent"],
            "agile": ["agile"],
            "noise": ["intermittent"],
            "mixed": ["periodic", "burst", "intermittent", "agile"],
            "shift": ["periodic", "burst", "intermittent", "agile"],
        }[self.scenario]

        n_sources = self.n_active_sources
        if self.scenario == "noise":
            n_sources = max(2, self.n_active_sources // 2)

        sources: List[ActivitySource] = []
        available = list(range(self.n_bands))
        rng.shuffle(available)
        cursor = 0
        for sid in range(n_sources):
            kind = kinds_for_scenario[sid % len(kinds_for_scenario)]
            if kind == "agile":
                width = int(rng.integers(2, 5))
                bands = [available[(cursor + i) % self.n_bands] for i in range(width)]
                cursor += width
            else:
                bands = [available[cursor % self.n_bands]]
                cursor += 1

            duty_scale = 0.25 if self.scenario == "noise" else 1.0
            sources.append(
                ActivitySource(
                    source_id=sid,
                    kind=kind,
                    bands=bands,
                    period=int(rng.integers(self.cycle_length // 4, self.cycle_length * 2)),
                    duty=max(1, int(rng.integers(4, max(5, self.cycle_length // 4)) * duty_scale)),
                    phase=int(rng.integers(0, self.cycle_length)),
                    p_on=float(rng.uniform(0.02, 0.09) * duty_scale),
                    p_off=float(rng.uniform(0.15, 0.40)),
                    hop_interval=int(rng.integers(20, 90)),
                    intensity=float(rng.uniform(0.7, 1.0)),
                )
            )
        return sources

    def _generate(self) -> None:
        rng = self._rng
        for start, end in self._segment_bounds():
            sources = self._draw_sources()
            self.segments.append(EnvironmentSegment(start=start, end=end, sources=sources))
            for src in sources:
                self._paint_source(src, start, end, rng)

        # Sparse background activity keeps unmodelled bands from being trivially empty.
        if self.base_activity > 0:
            background_rate = self.base_activity * (0.15 if self.scenario == "noise" else 0.05)
            background = rng.random((self.n_slots, self.n_bands)) < background_rate
            self.truth = np.logical_or(self.truth.astype(bool), background).astype(np.uint8)

    def _paint_source(self, src: ActivitySource, start: int, end: int, rng: np.random.Generator) -> None:
        length = end - start
        if length <= 0:
            return
        slots = np.arange(start, end)

        if src.kind == "periodic":
            on = ((slots + src.phase) % max(1, src.period)) < src.duty
            self.truth[start:end, src.bands[0]] |= on.astype(np.uint8)

        elif src.kind == "burst":
            active = np.zeros(length, dtype=bool)
            t = 0
            arrival_rate = max(1, src.period // 2)
            while t < length:
                gap = int(rng.exponential(arrival_rate)) + 1
                t += gap
                if t >= length:
                    break
                burst_len = max(1, int(rng.geometric(1.0 / max(2, src.duty))))
                active[t:min(length, t + burst_len)] = True
                t += burst_len
            self.truth[start:end, src.bands[0]] |= active.astype(np.uint8)

        elif src.kind == "intermittent":
            active = np.zeros(length, dtype=bool)
            state = rng.random() < 0.2
            draws = rng.random(length)
            for i in range(length):
                if state:
                    state = draws[i] >= src.p_off
                else:
                    state = draws[i] < src.p_on
                active[i] = state
            self.truth[start:end, src.bands[0]] |= active.astype(np.uint8)

        elif src.kind == "agile":
            # Activity hops between a small set of abstract bands over time.
            hop = max(1, src.hop_interval)
            on = ((slots + src.phase) % max(1, src.period)) < src.duty
            band_index = ((slots - start) // hop) % len(src.bands)
            for k, band in enumerate(src.bands):
                mask = on & (band_index == k)
                self.truth[start:end, band] |= mask.astype(np.uint8)

    # ------------------------------------------------------------- inspection
    def is_active(self, slot: int, band_id: int) -> bool:
        """Hidden truth lookup. Receiver simulator and evaluation engine only."""
        return bool(self.truth[slot, band_id])

    def active_bands(self, slot: int) -> np.ndarray:
        return np.flatnonzero(self.truth[slot])

    def activity_rate(self) -> float:
        return float(self.truth.mean())

    def bursts(self) -> List[Tuple[int, int, int]]:
        """Contiguous activity runs as ``(band_id, start_slot, end_slot_exclusive)``.

        Used by the evaluation engine to score intercept delay and the fraction
        of activity opportunities that were ever intercepted.
        """
        runs: List[Tuple[int, int, int]] = []
        for band in range(self.n_bands):
            column = self.truth[:, band]
            if not column.any():
                continue
            padded = np.concatenate(([0], column, [0]))
            edges = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            for s, e in zip(starts, ends):
                runs.append((band, int(s), int(e)))
        runs.sort(key=lambda r: r[1])
        return runs

    def describe(self) -> Dict[str, object]:
        return {
            "n_bands": self.n_bands,
            "n_slots": self.n_slots,
            "scenario": self.scenario,
            "seed": self.seed,
            "change_points": list(self.change_points),
            "activity_rate": round(self.activity_rate(), 4),
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "sources": [s.describe() for s in seg.sources],
                }
                for seg in self.segments
            ],
        }


__all__ = ["SyntheticEnvironment", "ActivitySource", "EnvironmentSegment", "SCENARIOS"]
