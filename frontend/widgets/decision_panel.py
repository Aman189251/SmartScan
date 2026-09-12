"""Decision explanation panel.

Answers one question for the operator: why this band, now.  Every number shown
is the value the scheduler actually used, so the panel is an audit trail rather
than a restatement of the outcome.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from .cards import Badge, Card, KeyValueGrid


class CandidateStrip(QWidget):
    """Candidate bands with their Thompson draws, selected one highlighted."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def update_candidates(self, candidates: Dict[str, float], selected: int) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        for band_str, score in ordered[:8]:
            band = int(band_str)
            chosen = band == selected
            chip = QLabel(f"B{band + 1:02d}  {score:.2f}")
            chip.setAlignment(Qt.AlignCenter)
            chip.setToolTip(
                "Selected: highest Thompson draw this slot." if chosen
                else "Considered, but drew lower than the selected band.")
            chip.setStyleSheet(theme.chip_style(
                theme.PRIMARY if chosen else theme.TEXT_DIM, strong=chosen))
            chip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self._layout.addWidget(chip)
        self._layout.addStretch(1)


class DecisionPanel(Card):
    """The 'why was this band selected' card."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(
            "Why was this band selected", parent,
            description="Every number here is the value the scheduler actually used, "
                        "not a restatement of what happened.")
        self.state_badge = Badge("IDLE", theme.TEXT_FAINT)
        self.add_header_widget(self.state_badge)

        header = QHBoxLayout()
        header.setSpacing(12)
        self.band_label = QLabel("--")
        self.band_label.setObjectName("StatValue")
        header.addWidget(self.band_label)

        self.outcome_badge = Badge("--", theme.TEXT_FAINT)
        self.outcome_badge.setMinimumWidth(150)
        header.addWidget(self.outcome_badge)
        header.addStretch(1)
        self.add_layout(header)

        self.grid = KeyValueGrid(columns=2)
        for key in ("Model probability", "Current belief", "Uncertainty",
                    "Thompson sample", "Alpha / Beta", "Time since scan",
                    "Recent detections", "Adaptive influence"):
            self.grid.add_row(key)
        self.add(self.grid)

        self.candidates = CandidateStrip()
        self.add(self.candidates)

        self.rationale = QLabel("Waiting for the first decision.")
        self.rationale.setWordWrap(True)
        self.rationale.setObjectName("Dim")
        self.rationale.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.rationale.setStyleSheet(
            f"background: {theme.rgba(theme.ACCENT, 0.07)};"
            f"border-left: 3px solid {theme.ACCENT};"
            f"border-radius: 6px; padding: 10px 12px; color: {theme.TEXT_DIM};"
        )
        self.add(self.rationale)
        # Keeps the rows packed at the top instead of spreading out whenever
        # the panel is given more height than its content needs.
        self.body().addStretch(1)

    def update_decision(self, status: Dict[str, Any], bands: Optional[Dict[str, Any]] = None,
                        since_scan: Optional[int] = None) -> None:
        current = status.get("current")
        state = str(status.get("state", "IDLE"))
        self.state_badge.set_state(
            state, theme.HIT if state == "RUNNING" else theme.TEXT_FAINT)
        if not current:
            return

        band = int(current.get("band", 0))
        self.band_label.setText(current.get("label", f"B{band + 1:02d}"))
        outcome = str(current.get("outcome", "--"))
        self.outcome_badge.set_state(
            outcome.replace("_", " ").title(),
            theme.OUTCOME_COLORS.get(outcome, theme.TEXT_FAINT))

        std = float(current.get("posterior_std", 0.0))
        self.grid.set("Model probability", f"{float(current.get('ml_probability', 0)):.3f}")
        self.grid.set("Current belief", f"{float(current.get('posterior_mean', 0)):.3f}")
        self.grid.set("Uncertainty", f"+/- {std:.3f}")
        self.grid.set("Thompson sample", f"{float(current.get('sampled', 0)):.3f}",
                      theme.ACCENT_DEEP)
        self.grid.set("Alpha / Beta",
                      f"{float(current.get('alpha', 1)):.2f} / {float(current.get('beta', 1)):.2f}")
        self.grid.set("Time since scan",
                      f"{since_scan} slots" if since_scan is not None else "--")

        hits = "--"
        if bands:
            hit_counts = bands.get("hits") or []
            scans = bands.get("scans") or []
            if band < len(hit_counts):
                hits = f"{int(hit_counts[band])} of {int(scans[band])} looks"
        self.grid.set("Recent detections", hits)

        influence = float(current.get("adaptive_influence", 0.0))
        self.grid.set("Adaptive influence", f"{influence:.0%}",
                      theme.WARN if influence > 0 else None)

        self.candidates.update_candidates(current.get("scores") or {}, band)
        self.rationale.setText(current.get("rationale") or "")


__all__ = ["DecisionPanel", "CandidateStrip"]
