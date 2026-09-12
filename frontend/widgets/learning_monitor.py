"""Learning monitor widgets: the adaptive state ladder and acceptance criteria."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from .cards import Badge, Card, KeyValueGrid

STATES: List[str] = [
    "OFFLINE_PRIOR",
    "LEARNING",
    "SHADOW",
    "ADAPTIVE_ASSISTED",
    "ADAPTIVE_CONTROL",
]

STATE_CAPTIONS = {
    "OFFLINE_PRIOR": "Offline model sets the starting belief",
    "LEARNING": "Live observations update the Bayesian state",
    "SHADOW": "Parallel recommendations, no control",
    "ADAPTIVE_ASSISTED": "Bounded influence on the decision layer",
    "ADAPTIVE_CONTROL": "Approved level of control granted",
}


class StateLadder(QWidget):
    """Five-step promotion ladder with the current step lit."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._steps: Dict[str, QLabel] = {}
        for index, state in enumerate(STATES):
            label = QLabel(f"{index + 1}  {state.replace('_', ' ').title()}")
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(STATE_CAPTIONS[state])
            label.setWordWrap(True)
            self._steps[state] = label
            layout.addWidget(label, 1)
        self.set_state("OFFLINE_PRIOR")

    def set_state(self, current: str) -> None:
        try:
            current_index = STATES.index(current)
        except ValueError:
            current_index = 0
        for index, state in enumerate(STATES):
            label = self._steps[state]
            if index < current_index:
                color, background, weight = theme.TEXT_DIM, theme.rgba(theme.TEXT_DIM, 0.08), "500"
            elif index == current_index:
                color = theme.ADAPTIVE_COLORS.get(state, theme.ACCENT)
                background, weight = theme.rgba(color, 0.15), "700"
            else:
                color, background, weight = theme.TEXT_FAINT, "transparent", "400"
            label.setStyleSheet(
                f"color: {color}; background: {background};"
                f"border: 1px solid {theme.rgba(color, 0.27)};"
                f"border-radius: 7px; padding: 8px 6px; font-size: 11px; font-weight: {weight};"
            )


class AcceptanceTable(QTableWidget):
    """Acceptance criteria with measured value, threshold and pass state."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Criterion", "Measured", "Threshold", "Status"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMinimumHeight(190)

    def update_criteria(self, checks: Dict[str, Dict[str, Any]]) -> None:
        labels = {
            "observations": "Observations collected",
            "brier": "Recent Brier score (lower is better)",
            "calibration_error": "Calibration error",
            "advantage": "Advantage over offline model",
            "no_drift": "No active drift alarm",
        }
        self.setRowCount(len(labels))
        for row, (key, label) in enumerate(labels.items()):
            entry = checks.get(key, {})
            value = entry.get("value")
            threshold = entry.get("threshold")
            passed = bool(entry.get("passed"))

            self.setItem(row, 0, QTableWidgetItem(label))
            self.setItem(row, 1, QTableWidgetItem(_fmt(value)))
            self.setItem(row, 2, QTableWidgetItem(_fmt(threshold)))
            self.setItem(row, 3, QTableWidgetItem("PASS" if passed else "PENDING"))
            for column in range(4):
                item = self.item(row, column)
                if item is None:
                    continue
                if column == 3:
                    item.setForeground(_brush(theme.HIT if passed else theme.TEXT_FAINT))
                elif column == 0:
                    item.setForeground(_brush(theme.TEXT))
                else:
                    item.setForeground(_brush(theme.TEXT_DIM))


class LearningMonitorCard(Card):
    """Adaptive layer status, monitors and drift indicator."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(
            "Hidden adaptive learner", parent,
            description="Runs continuously in the background. It starts with zero "
                        "influence and only earns more by passing its criteria.")
        self.state_badge = Badge("OFFLINE PRIOR", theme.TEXT_FAINT)
        self.add_header_widget(self.state_badge)
        self.drift_badge = Badge("stable", theme.HIT)
        self.add_header_widget(self.drift_badge)

        self.ladder = StateLadder()
        self.add(self.ladder)

        self.grid = KeyValueGrid(columns=2)
        for key in ("Adaptive influence", "Observations", "Drift alarms",
                    "Shadow recommendation", "Offline Brier", "Adaptive Brier",
                    "Offline accuracy", "Adaptive accuracy"):
            self.grid.add_row(key)
        self.add(self.grid)

    def update_status(self, status: Dict[str, Any]) -> None:
        state = str(status.get("state", "OFFLINE_PRIOR"))
        self.ladder.set_state(state)
        self.state_badge.set_state(state.replace("_", " "),
                                   theme.ADAPTIVE_COLORS.get(state, theme.TEXT_FAINT))
        drift = bool(status.get("drift_flag"))
        # Deliberately hedged: the alarm means recent evidence disagrees with
        # confident belief, which is a strong hint of change but not proof of one.
        self.drift_badge.set_state("belief contradicted" if drift else "belief holding",
                                   theme.WARN if drift else theme.HIT)
        self.drift_badge.setToolTip(
            "Raised when recent looks at bands the system was confident about stop "
            "agreeing with that belief. Evidence is discounted faster and adaptive "
            "influence is withdrawn until it clears.")

        offline = status.get("offline", {})
        adaptive = status.get("adaptive", {})
        influence = float(status.get("influence", 0.0))
        recommendation = status.get("recommendation")

        self.grid.set("Adaptive influence", f"{influence:.0%}",
                      theme.WARN if influence > 0 else None)
        self.grid.set("Observations", str(status.get("observations", 0)))
        self.grid.set("Drift alarms", str(status.get("drift_count", 0)),
                      theme.WARN if status.get("drift_count") else None)
        self.grid.set("Shadow recommendation",
                      f"B{int(recommendation) + 1:02d}" if recommendation is not None else "--")
        self.grid.set("Offline Brier", _fmt(offline.get("brier")))
        self.grid.set("Adaptive Brier", _fmt(adaptive.get("brier")),
                      theme.HIT if _lower(adaptive.get("brier"), offline.get("brier")) else None)
        self.grid.set("Offline accuracy", _pct(offline.get("accuracy")))
        self.grid.set("Adaptive accuracy", _pct(adaptive.get("accuracy")))


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: Any) -> str:
    return "--" if value is None else f"{float(value) * 100:.1f}%"


def _lower(a: Any, b: Any) -> bool:
    try:
        return float(a) < float(b)
    except (TypeError, ValueError):
        return False


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color))


__all__ = ["LearningMonitorCard", "StateLadder", "AcceptanceTable", "STATES"]
