"""Learning monitor view: offline model status and the hidden adaptive layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..visualization.charts import CurveChart
from ..widgets.cards import Badge, Card, KeyValueGrid, PanelMenu, ViewHeading
from ..widgets.learning_monitor import AcceptanceTable, LearningMonitorCard


class LearningView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(14)

        self.panel_menu = PanelMenu()
        toolbar = QHBoxLayout()
        toolbar.addWidget(ViewHeading(
            "Learning monitor",
            "What the offline model knows, and how much the background learner is "
            "currently trusted."))
        toolbar.addStretch(1)
        toolbar.addWidget(self.panel_menu)
        layout.addLayout(toolbar)

        top = QHBoxLayout()
        top.setSpacing(14)

        self.model_card = Card(
            "Offline model",
            description="Provenance and held-out scores for the model loaded right now.")
        self.model_badge = Badge("no model", theme.WARN)
        self.model_card.add_header_widget(self.model_badge)
        self.model_grid = KeyValueGrid()
        for key in ("Model", "Version", "Trained", "Features", "Dataset",
                    "Test ROC-AUC", "Test Brier", "Calibration error"):
            self.model_grid.add_row(key)
        self.model_card.add(self.model_grid)
        self.model_note = QLabel("")
        self.model_note.setObjectName("Faint")
        self.model_note.setWordWrap(True)
        self.model_card.add(self.model_note)
        top.addWidget(self.model_card, 1)

        self.monitor = LearningMonitorCard()
        top.addWidget(self.monitor, 2)
        layout.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(14)

        self.acceptance_card = Card(
            "Acceptance criteria for adaptive influence",
            description="Promotion needs every criterion to hold for several consecutive "
                        "evaluations. A drift alarm withdraws influence immediately.")
        self.acceptance = AcceptanceTable()
        self.acceptance_card.add(self.acceptance, 1)
        self.streak_label = QLabel("--")
        self.streak_label.setObjectName("Dim")
        self.acceptance_card.add(self.streak_label)
        middle.addWidget(self.acceptance_card, 1)

        self.quality_card = Card(
            "Recent prediction quality",
            description="Offline model against the adaptive layer, both scored on the "
                        "scans that were actually executed.")
        self.quality_curve = CurveChart("Brier score (lower is better)")
        self.quality_card.add(self.quality_curve, 1)
        middle.addWidget(self.quality_card, 1)
        layout.addLayout(middle)

        self.events_card = Card(
            "Adaptive events",
            description="Promotions, demotions and drift alarms, newest first.")
        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels(["Slot", "Event", "Detail"])
        self.events.verticalHeader().setVisible(False)
        self.events.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events.horizontalHeader().setStretchLastSection(True)
        self.events.setColumnWidth(0, 80)
        self.events.setColumnWidth(1, 170)
        self.events.setMinimumHeight(180)
        self.events_card.add(self.events, 1)
        layout.addWidget(self.events_card)

        self.panel_menu.add_panels([
            self.model_card, self.monitor, self.acceptance_card,
            self.quality_card, self.events_card,
        ])

        self._offline_history: List[float] = []
        self._adaptive_history: List[float] = []
        self._event_rows: List[tuple] = []

    # ----------------------------------------------------------------- update
    def update_model(self, models: Dict[str, Any]) -> None:
        loaded = models.get("loaded", {}) or {}
        trained = bool(loaded.get("trained"))
        self.model_badge.set_state(
            "trained model loaded" if trained else "no trained model",
            theme.HIT if trained else theme.WARN)

        metrics = loaded.get("validation_metrics", {}) or {}
        calibration = loaded.get("calibration_metrics", {}) or {}
        self.model_grid.set("Model", str(loaded.get("kind", "--")))
        self.model_grid.set("Version", str(loaded.get("version", "--")))
        self.model_grid.set("Trained", str(loaded.get("trained_at", "--"))[:19].replace("T", " "))
        self.model_grid.set("Features", str(loaded.get("feature_version", "--")))
        self.model_grid.set("Dataset", str(loaded.get("dataset_version", "--")))
        self.model_grid.set("Test ROC-AUC", _fmt(metrics.get("roc_auc")))
        self.model_grid.set("Test Brier", _fmt(metrics.get("brier")))
        self.model_grid.set("Calibration error",
                            _fmt(calibration.get("expected_calibration_error")))
        self.model_note.setText(
            loaded.get("note", "") or
            "The offline model supplies prior knowledge only. It never selects a band by itself.")

    def update_adaptive(self, status: Dict[str, Any]) -> None:
        self.monitor.update_status(status)
        acceptance = status.get("acceptance", {}) or {}
        self.acceptance.update_criteria(acceptance.get("checks", {}) or {})
        streak = acceptance.get("streak", 0)
        window = acceptance.get("stability_window", 0)
        passed = acceptance.get("all_passed")
        self.streak_label.setText(
            f"Consecutive passing evaluations: {streak} of {window} required"
            + ("  -  all criteria currently met" if passed else ""))

        offline = (status.get("offline") or {}).get("brier")
        adaptive = (status.get("adaptive") or {}).get("brier")
        if offline is not None:
            self._offline_history.append(float(offline))
            self._adaptive_history.append(float(adaptive if adaptive is not None else offline))
            if len(self._offline_history) > 400:
                self._offline_history = self._offline_history[-400:]
                self._adaptive_history = self._adaptive_history[-400:]
            self.quality_curve.set_series(
                "offline", self._offline_history, theme.ACCENT_DEEP,
                label="Offline model",
                note="The trained model's Brier score on the scans actually executed.")
            self.quality_curve.set_series(
                "adaptive", self._adaptive_history, theme.HIT,
                label="Adaptive layer",
                note="The background learner's Brier score on the same scans. Below "
                     "the blue line means it is currently the better predictor.")

    def add_events(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            row = (event.get("slot"), event.get("event_type"), event.get("detail"))
            if row in self._event_rows:
                continue
            self._event_rows.append(row)
        self._event_rows = self._event_rows[-100:]

        self.events.setRowCount(len(self._event_rows))
        for index, (slot, kind, detail) in enumerate(reversed(self._event_rows)):
            color = theme.WARN if kind == "DRIFT_DETECTED" else theme.ACCENT_DEEP
            for column, text in enumerate((str(slot), str(kind), str(detail))):
                item = QTableWidgetItem(text)
                if column == 1:
                    item.setForeground(QBrush(QColor(color)))
                self.events.setItem(index, column, item)

    def reset(self) -> None:
        self._offline_history.clear()
        self._adaptive_history.clear()
        self._event_rows.clear()
        self.events.setRowCount(0)
        self.quality_curve.clear()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if number != number else f"{number:.{digits}f}"


__all__ = ["LearningView"]
