"""Receiver simulator.

Converts hidden environment state into an imperfect observation and enforces
the sensing constraint that motivates the whole project: the receiver can look
at only a small part of the search space at a time.

    ACTIVE   -> HIT with probability Pd, otherwise MISS
    INACTIVE -> FALSE_ALARM with probability Pfa, otherwise CORRECT_NON_DETECTION

Pd and Pfa are simulation parameters. They are not equipment specifications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.types import NormalizedObservation, Outcome, ScanCommand


@dataclass(slots=True)
class ReceiverConstraints:
    """Declared capability of the modelled receiver."""

    n_bands: int
    capacity: int = 1            # bands observable within one slot
    scan_duration: int = 1       # slots consumed by a single scan
    revisit_lockout: int = 0     # slots a band stays unavailable after a scan
    tuning_penalty: int = 0      # extra slots when moving far across the search space

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_bands": self.n_bands,
            "capacity": self.capacity,
            "scan_duration": self.scan_duration,
            "revisit_lockout": self.revisit_lockout,
            "tuning_penalty": self.tuning_penalty,
        }


@dataclass(slots=True)
class ReceiverStatus:
    state: str = "READY"
    current_band: Optional[int] = None
    busy_until: int = -1
    scans_executed: int = 0
    detections: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "current_band": self.current_band,
            "busy_until": self.busy_until,
            "scans_executed": self.scans_executed,
            "detections": self.detections,
            "last_error": self.last_error,
        }


class ReceiverSimulator:
    """Imperfect, capacity-limited observer over a hidden environment."""

    def __init__(
        self,
        environment,
        pd: float = 0.90,
        pfa: float = 0.03,
        capacity: int = 1,
        scan_duration: int = 1,
        revisit_lockout: int = 0,
        seed: int = 7,
    ):
        if not 0.0 <= pd <= 1.0 or not 0.0 <= pfa <= 1.0:
            raise ValueError("pd and pfa must lie in [0, 1]")
        self.environment = environment
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.constraints = ReceiverConstraints(
            n_bands=environment.n_bands,
            capacity=int(capacity),
            scan_duration=int(scan_duration),
            revisit_lockout=int(revisit_lockout),
        )
        self.status = ReceiverStatus()
        self._rng = np.random.default_rng(seed)
        self._last_scanned: Dict[int, int] = {}
        self._history: List[NormalizedObservation] = []
        self._current_slot = -1
        self._scans_in_slot = 0

    # ------------------------------------------------------------- constraints
    def available_bands(self, slot: int) -> List[int]:
        """Bands the receiver is permitted to tune to at ``slot``."""
        lockout = self.constraints.revisit_lockout
        if lockout <= 0:
            return list(range(self.constraints.n_bands))
        return [
            b for b in range(self.constraints.n_bands)
            if slot - self._last_scanned.get(b, -10**9) > lockout
        ]

    def validate(self, command: ScanCommand) -> None:
        """Raise if a scan command violates a declared receiver constraint."""
        if not 0 <= command.band_id < self.constraints.n_bands:
            raise ValueError(f"band {command.band_id} outside receiver range")
        # Instantaneous bandwidth limit: at most ``capacity`` looks per slot, and
        # no look at all while a longer scan is still in progress.
        in_slot = self._scans_in_slot if command.slot == self._current_slot else 0
        if in_slot >= self.constraints.capacity:
            raise RuntimeError(
                f"receiver capacity {self.constraints.capacity} exhausted at slot {command.slot}"
            )
        if in_slot == 0 and command.slot < self.status.busy_until:
            raise RuntimeError(
                f"receiver busy until slot {self.status.busy_until}, command at {command.slot}"
            )
        lockout = self.constraints.revisit_lockout
        if lockout > 0 and command.slot - self._last_scanned.get(command.band_id, -10**9) <= lockout:
            raise RuntimeError(f"band {command.band_id} in revisit lockout at slot {command.slot}")

    # ------------------------------------------------------------------ observe
    def observe(self, command: ScanCommand, run_id: str = "run", source_id: str = "simulator") -> NormalizedObservation:
        """Execute one scan and return a normalised observation."""
        self.validate(command)
        slot, band = command.slot, command.band_id
        active = self.environment.is_active(slot, band)
        draw = float(self._rng.random())

        if active:
            outcome = Outcome.HIT if draw < self.pd else Outcome.MISS
        else:
            outcome = Outcome.FALSE_ALARM if draw < self.pfa else Outcome.CORRECT_NON_DETECTION

        detected = outcome.is_detection
        # Confidence is a normalised, receiver-side quality proxy. It never
        # exposes truth: it is derived from the reported detection only.
        quality = 0.5 + 0.5 * abs(draw - (self.pd if detected else self.pfa))

        if slot == self._current_slot:
            self._scans_in_slot += 1
        else:
            self._current_slot = slot
            self._scans_in_slot = 1

        self.status.scans_executed += 1
        self.status.detections += int(detected)
        self.status.current_band = band
        self.status.busy_until = slot + max(1, self.constraints.scan_duration)
        self.status.state = "READY"
        self._last_scanned[band] = slot

        observation = NormalizedObservation(
            run_id=run_id,
            source_id=source_id,
            slot=slot,
            timestamp=time.time(),
            band_id=band,
            scan_duration=max(1, self.constraints.scan_duration),
            outcome=outcome,
            detected=detected,
            quality=round(float(min(1.0, max(0.0, quality))), 4),
            receiver_state=self.status.state,
            metadata={"dwell": command.dwell},
        )
        self._history.append(observation)
        return observation

    # --------------------------------------------------------------- utilities
    def reset(self, seed: Optional[int] = None) -> None:
        self.status = ReceiverStatus()
        self._last_scanned.clear()
        self._history.clear()
        self._current_slot = -1
        self._scans_in_slot = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def history(self) -> List[NormalizedObservation]:
        return self._history

    def describe(self) -> Dict[str, object]:
        return {
            "pd": self.pd,
            "pfa": self.pfa,
            "constraints": self.constraints.to_dict(),
            "status": self.status.to_dict(),
        }


__all__ = ["ReceiverSimulator", "ReceiverConstraints", "ReceiverStatus"]
