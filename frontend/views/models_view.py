"""Models and runs view: registry, training, feature importance, run history."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..visualization.charts import ImportanceChart
from ..widgets.cards import Card, PanelMenu, ToggleSwitch, ViewHeading


class ModelsView(QWidget):
    train_requested = Signal(dict)
    promote_requested = Signal(str)
    refresh_requested = Signal()

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
            "Models and runs",
            "Train an offline model, see what it leans on, and review what past runs did."))
        toolbar.addStretch(1)
        toolbar.addWidget(self.panel_menu)
        layout.addLayout(toolbar)

        self.training_card = self._build_training_card()
        layout.addWidget(self.training_card)

        middle = QHBoxLayout()
        middle.setSpacing(14)

        self.registry_card = Card(
            "Model registry",
            description="Every trained model. Select one to see its feature importance, "
                        "or promote it for the next run.")
        self.registry_table = QTableWidget(0, 6)
        self.registry_table.setHorizontalHeaderLabels(
            ["Model id", "Kind", "Trained", "ROC-AUC", "Brier", "Production"])
        self.registry_table.verticalHeader().setVisible(False)
        self.registry_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.registry_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.registry_table.setAlternatingRowColors(True)
        self.registry_table.horizontalHeader().setStretchLastSection(True)
        self.registry_table.itemSelectionChanged.connect(self._selection_changed)
        self.registry_card.add(self.registry_table, 1)

        registry_actions = QHBoxLayout()
        registry_actions.addStretch(1)
        self.promote_button = QPushButton("Promote to production")
        self.promote_button.setEnabled(False)
        self.promote_button.clicked.connect(self._emit_promote)
        registry_actions.addWidget(self.promote_button)
        self.registry_card.add_layout(registry_actions)
        middle.addWidget(self.registry_card, 3)

        self.importance_card = Card(
            "Feature importance",
            description="What the selected model leans on. Built from observation "
                        "history only - ground truth is never a feature.")
        self.importance_chart = ImportanceChart()
        self.importance_card.add(self.importance_chart, 1)
        self.importance_table = QTableWidget(0, 2)
        self.importance_table.setHorizontalHeaderLabels(["Feature", "Weight"])
        self.importance_table.verticalHeader().setVisible(False)
        self.importance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.importance_table.horizontalHeader().setStretchLastSection(True)
        self.importance_table.setMaximumHeight(150)
        self.importance_card.add(self.importance_table)
        middle.addWidget(self.importance_card, 2)
        layout.addLayout(middle, 1)

        self.runs_card = Card(
            "Recent runs",
            description="The last runs recorded in the database, with their headline metrics.")
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(
            ["Run", "Strategy", "Seed", "Status", "Intercept rate", "Pd", "Reward"])
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_card.add(self.runs_table, 1)
        layout.addWidget(self.runs_card, 1)

        self.panel_menu.add_panels([
            self.training_card, self.registry_card, self.importance_card, self.runs_card,
        ])

        self._selected_model: Optional[str] = None

    # ---------------------------------------------------------------- training
    def _build_training_card(self) -> Card:
        card = Card(
            "Offline training",
            description="Generates seeded traces across every scenario, builds features "
                        "from observation history, then trains XGBoost with a Random "
                        "Forest baseline under the same split.")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(_dim("Runs per scenario"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 20)
        self.runs_spin.setValue(3)
        row.addWidget(self.runs_spin)

        row.addWidget(_dim("Base seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999_999)
        self.seed_spin.setValue(4242)
        row.addWidget(self.seed_spin)

        row.addWidget(_dim("Bands sampled per slot"))
        self.bands_spin = QSpinBox()
        self.bands_spin.setRange(1, 32)
        self.bands_spin.setValue(6)
        row.addWidget(self.bands_spin)

        row.addSpacing(6)
        self.rf_toggle = ToggleSwitch(checked=True)
        self.rf_toggle.setToolTip(
            "Trains a Random Forest on the identical split, as a sanity baseline "
            "for the XGBoost numbers.")
        row.addWidget(self.rf_toggle)
        rf_label = _dim("Random Forest baseline")
        rf_label.setToolTip(self.rf_toggle.toolTip())
        row.addWidget(rf_label)

        row.addStretch(1)
        self.train_button = QPushButton("Generate dataset and train")
        self.train_button.setObjectName("Primary")
        self.train_button.clicked.connect(self._emit_train)
        row.addWidget(self.train_button)
        card.add_layout(row)

        status_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setMaximum(1000)
        self.progress.setTextVisible(False)
        status_row.addWidget(self.progress, 1)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("Faint")
        self.status_label.setMinimumWidth(340)
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self.status_label)
        card.add_layout(status_row)
        return card

    def _emit_train(self) -> None:
        self.train_requested.emit({
            "runs_per_scenario": int(self.runs_spin.value()),
            "base_seed": int(self.seed_spin.value()),
            "bands_per_slot": int(self.bands_spin.value()),
            "train_random_forest": bool(self.rf_toggle.isChecked()),
        })

    def set_busy(self, busy: bool, message: str = "", progress: float = 0.0) -> None:
        self.train_button.setEnabled(not busy)
        self.train_button.setText("Training..." if busy else "Generate dataset and train")
        self.progress.setValue(int(progress * 1000))
        if message:
            self.status_label.setText(message)

    # ---------------------------------------------------------------- registry
    def set_models(self, payload: Dict[str, Any]) -> None:
        rows: List[Dict[str, Any]] = payload.get("registry", []) or []
        self.registry_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            metrics = row.get("validation_metrics", {}) or {}
            values = [
                row.get("model_id", ""),
                row.get("kind", ""),
                str(row.get("trained_at", ""))[:19].replace("T", " "),
                _fmt(metrics.get("roc_auc")),
                _fmt(metrics.get("brier")),
                "production" if row.get("is_latest") else "",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 5 and text:
                    item.setForeground(QBrush(QColor(theme.HIT)))
                self.registry_table.setItem(index, column, item)
        self.registry_table.resizeColumnsToContents()

    def set_importance(self, importance: Dict[str, float]) -> None:
        self.importance_chart.update_importance(importance)
        self.importance_table.setRowCount(len(importance))
        for index, (name, weight) in enumerate(importance.items()):
            self.importance_table.setItem(index, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(f"{float(weight):.4f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.importance_table.setItem(index, 1, item)

    def set_runs(self, runs: List[Dict[str, Any]]) -> None:
        self.runs_table.setRowCount(len(runs))
        for index, run in enumerate(runs):
            summary = run.get("summary", {}) or {}
            values = [
                str(run.get("run_id", ""))[:24],
                str(run.get("strategy", "")),
                str(run.get("seed", "")),
                str(run.get("status", "")),
                _fmt(summary.get("average_intercept_rate")),
                _fmt(summary.get("probability_of_detection")),
                _fmt(summary.get("cumulative_reward"), 1),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column >= 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.runs_table.setItem(index, column, item)
        self.runs_table.resizeColumnsToContents()

    def _selection_changed(self) -> None:
        items = self.registry_table.selectedItems()
        if not items:
            self._selected_model = None
            self.promote_button.setEnabled(False)
            return
        row = items[0].row()
        cell = self.registry_table.item(row, 0)
        self._selected_model = cell.text() if cell else None
        self.promote_button.setEnabled(self._selected_model is not None)
        if self._selected_model:
            self.refresh_requested.emit()

    def selected_model(self) -> Optional[str]:
        return self._selected_model

    def _emit_promote(self) -> None:
        if self._selected_model:
            self.promote_requested.emit(self._selected_model)


def _dim(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Dim")
    return label


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if number != number else f"{number:.{digits}f}"


__all__ = ["ModelsView"]
