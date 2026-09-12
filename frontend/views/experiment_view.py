"""Experiment Arena: run every policy on identical environments and compare."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
from ..visualization.charts import ComparisonChart
from ..widgets.cards import Card, PanelMenu, ToggleSwitch, ViewHeading
from ..widgets.chart_legend import ChartLegend

COMPARE_METRICS = [
    ("average_intercept_rate", "Average intercept rate"),
    ("intercept_ratio", "Fraction of activity intercepted"),
    ("probability_of_detection", "Measured Pd"),
    ("probability_of_false_alarm", "Measured Pfa"),
    ("average_intercept_delay", "Average intercept delay (lower is better)"),
    ("scan_efficiency", "Scan efficiency"),
    ("coverage", "Coverage"),
    ("cumulative_reward", "Cumulative reward"),
    ("percentage_correct_predictions", "Correct predictions"),
]

#: Metrics where the winner is the smallest number, not the largest.
LOWER_IS_BETTER = {"average_intercept_delay", "probability_of_false_alarm"}

TABLE_METRICS = [
    ("average_intercept_rate", "Intercept rate"),
    ("intercept_ratio", "Intercepted"),
    ("average_intercept_delay", "Delay"),
    ("probability_of_detection", "Pd"),
    ("probability_of_false_alarm", "Pfa"),
    ("scan_efficiency", "Efficiency"),
    ("coverage", "Coverage"),
    ("cumulative_reward", "Reward"),
]

#: Columns the table shows by default; the rest are switched on from the menu.
DEFAULT_COLUMNS = {"average_intercept_rate", "intercept_ratio",
                   "average_intercept_delay", "probability_of_detection",
                   "probability_of_false_alarm"}


class ExperimentView(QWidget):
    run_requested = Signal(dict)
    load_requested = Signal(str)

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
            "Experiment arena",
            "Every policy on the identical seeded environment, so the difference "
            "is the policy and nothing else."))
        toolbar.addStretch(1)
        toolbar.addWidget(self.panel_menu)
        layout.addLayout(toolbar)

        self.setup_card = self._build_controls()
        layout.addWidget(self.setup_card)

        body = QHBoxLayout()
        body.setSpacing(14)

        self.chart_card = Card(
            "Comparison",
            description="One metric at a time, averaged across seeds. Whiskers show "
                        "how much the seeds disagreed.")
        metric_row = QHBoxLayout()
        metric_label = QLabel("Metric")
        metric_label.setObjectName("Dim")
        metric_row.addWidget(metric_label)
        self.metric_combo = _compact_combo(18)
        for key, label in COMPARE_METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._redraw)
        metric_row.addWidget(self.metric_combo, 1)
        self.chart_card.add_layout(metric_row)
        self.chart = ComparisonChart()
        self.chart_card.add(self.chart, 1)
        body.addWidget(self.chart_card, 1)

        self.table_card = Card(
            "Results",
            description="Every metric for every strategy. Use the column menu to "
                        "show only the ones you care about.")
        self.column_menu = PanelMenu("Columns")
        self.column_menu.setToolTip("Choose which metric columns the table shows")
        self.table_card.add_header_widget(self.column_menu)

        self.table = QTableWidget(0, len(TABLE_METRICS) + 1)
        self.table.setHorizontalHeaderLabels(
            ["Strategy"] + [label for _, label in TABLE_METRICS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_card.add(self.table, 1)

        self.table_legend = ChartLegend([
            ("Best value in this column", theme.PRIMARY, "bar",
             "Largest or smallest, whichever wins that metric."),
            ("The complete Smart Scan strategy", theme.ACCENT_DEEP, "bar", ""),
        ])
        self.table_card.add(self.table_legend)
        body.addWidget(self.table_card, 1)
        layout.addLayout(body, 1)

        self._result: Dict[str, Any] = {}
        self._build_column_menu()
        self.panel_menu.add_panels([self.setup_card, self.chart_card, self.table_card])

    # ---------------------------------------------------------------- controls
    def _build_controls(self) -> Card:
        card = Card(
            "Experiment setup",
            description="Pick the scenario, the seeds and how many slots each policy gets.")
        row = QHBoxLayout()
        row.setSpacing(10)

        row.addWidget(_dim("Scenario"))
        self.scenario_combo = _compact_combo(16)
        row.addWidget(self.scenario_combo)

        row.addWidget(_dim("Seeds"))
        self.seeds_edit = QLineEdit("101, 202, 303")
        self.seeds_edit.setMaximumWidth(160)
        row.addWidget(self.seeds_edit)

        row.addWidget(_dim("Slots"))
        self.slots_spin = QSpinBox()
        self.slots_spin.setRange(100, 20000)
        self.slots_spin.setSingleStep(100)
        self.slots_spin.setValue(1200)
        row.addWidget(self.slots_spin)

        row.addSpacing(6)
        self.ablation_toggle = ToggleSwitch()
        self.ablation_toggle.setToolTip(
            "Compare the complete strategy against versions with the adaptive layer, "
            "the ML prior or the mathematical layer removed.")
        row.addWidget(self.ablation_toggle)
        ablation_label = _dim("Layer ablation")
        ablation_label.setToolTip(self.ablation_toggle.toolTip())
        row.addWidget(ablation_label)

        row.addStretch(1)

        self.run_button = QPushButton("Run experiment")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._emit_run)
        row.addWidget(self.run_button)
        card.add_layout(row)

        status_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setMaximum(1000)
        self.progress.setTextVisible(False)
        status_row.addWidget(self.progress, 1)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("Faint")
        self.status_label.setMinimumWidth(320)
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self.status_label)
        card.add_layout(status_row)

        history_row = QHBoxLayout()
        history_row.addWidget(_dim("Saved results"))
        self.history_combo = _compact_combo(20)
        history_row.addWidget(self.history_combo, 1)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self._emit_load)
        history_row.addWidget(load_button)
        card.add_layout(history_row)
        return card

    def _build_column_menu(self) -> None:
        """Let the operator switch metric columns on and off."""
        menu = self.column_menu.menu()
        menu.clear()
        menu.addAction("Show every column", self._show_all_columns)
        menu.addSeparator()
        self._column_actions = {}
        for index, (key, label) in enumerate(TABLE_METRICS, start=1):
            action = menu.addAction(label)
            action.setCheckable(True)
            visible = key in DEFAULT_COLUMNS
            action.setChecked(visible)
            self.table.setColumnHidden(index, not visible)
            action.toggled.connect(
                lambda checked, column=index: self.table.setColumnHidden(column, not checked))
            self._column_actions[key] = action

    def _show_all_columns(self) -> None:
        for action in self._column_actions.values():
            action.setChecked(True)

    def set_scenarios(self, scenarios: List[Dict[str, Any]]) -> None:
        current = self.scenario_combo.currentData()
        self.scenario_combo.clear()
        for scenario in scenarios:
            self.scenario_combo.addItem(scenario.get("title", scenario["key"]), scenario["key"])
        if current:
            index = self.scenario_combo.findData(current)
            if index >= 0:
                self.scenario_combo.setCurrentIndex(index)

    def set_history(self, files: List[Dict[str, Any]]) -> None:
        current = self.history_combo.currentText()
        self.history_combo.clear()
        for entry in files:
            self.history_combo.addItem(entry["name"])
        if current:
            index = self.history_combo.findText(current)
            if index >= 0:
                self.history_combo.setCurrentIndex(index)

    def _emit_run(self) -> None:
        seeds = []
        for chunk in self.seeds_edit.text().replace(";", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                try:
                    seeds.append(int(chunk))
                except ValueError:
                    continue
        self.run_requested.emit({
            "scenario": self.scenario_combo.currentData() or "mixed",
            "seeds": seeds or [101, 202, 303],
            "max_slots": int(self.slots_spin.value()),
            "ablation": bool(self.ablation_toggle.isChecked()),
        })

    def _emit_load(self) -> None:
        name = self.history_combo.currentText()
        if name:
            self.load_requested.emit(name)

    # ----------------------------------------------------------------- results
    def set_busy(self, busy: bool, message: str = "", progress: float = 0.0) -> None:
        self.run_button.setEnabled(not busy)
        self.run_button.setText("Running..." if busy else "Run experiment")
        self.progress.setValue(int(progress * 1000))
        if message:
            self.status_label.setText(message)

    def set_result(self, result: Dict[str, Any]) -> None:
        self._result = result or {}
        self._redraw()
        self._fill_table()
        name = self._result.get("name", "")
        seeds = self._result.get("seeds", [])
        self.status_label.setText(f"{name}  -  {len(seeds)} seeds" if name else "Idle")

    def _labels(self) -> Dict[str, str]:
        arms = self._result.get("arms") or {}
        if arms:
            return {k: v for k, v in arms.items()}
        return {}

    def _redraw(self) -> None:
        aggregate = self._result.get("aggregate") or {}
        if not aggregate:
            return
        metric = self.metric_combo.currentData()
        labels, values, errors = [], [], []
        arm_labels = self._labels()
        for strategy, stats in aggregate.items():
            mean = stats.get(f"{metric}_mean")
            if mean is None:
                continue
            labels.append(arm_labels.get(strategy, strategy.replace("_", " ")))
            values.append(float(mean))
            errors.append(float(stats.get(f"{metric}_std", 0.0)))
        lower_is_better = metric in LOWER_IS_BETTER
        order = sorted(range(len(values)), key=lambda i: values[i],
                       reverse=not lower_is_better)
        self.chart.update_bars(
            [labels[i] for i in order], [values[i] for i in order],
            [errors[i] for i in order],
            metric=dict(COMPARE_METRICS).get(metric, metric),
            lower_is_better=lower_is_better,
        )

    def _fill_table(self) -> None:
        aggregate = self._result.get("aggregate") or {}
        arm_labels = self._labels()
        rows = sorted(
            aggregate.items(),
            key=lambda kv: kv[1].get("average_intercept_rate_mean", 0.0),
            reverse=True,
        )
        self.table.setRowCount(len(rows))
        best_values = {}
        for key, _ in TABLE_METRICS:
            candidates = [stats.get(f"{key}_mean") for _, stats in rows
                          if stats.get(f"{key}_mean") is not None]
            if candidates:
                best_values[key] = (min(candidates) if key in LOWER_IS_BETTER
                                    else max(candidates))

        for row, (strategy, stats) in enumerate(rows):
            name = QTableWidgetItem(arm_labels.get(strategy, strategy.replace("_", " ")))
            if strategy in ("smart_scan", "full"):
                name.setForeground(QBrush(QColor(theme.ACCENT_DEEP)))
                bold = QFont()
                bold.setBold(True)
                name.setFont(bold)
            self.table.setItem(row, 0, name)
            for column, (key, _) in enumerate(TABLE_METRICS, start=1):
                mean = stats.get(f"{key}_mean")
                std = stats.get(f"{key}_std")
                text = "--" if mean is None else f"{mean:.3f}"
                if mean is not None and std:
                    text += f" ±{std:.2f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if mean is not None and best_values.get(key) == mean:
                    item.setForeground(QBrush(QColor(theme.PRIMARY)))
                    bold = QFont()
                    bold.setBold(True)
                    item.setFont(bold)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()


def _dim(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Dim")
    return label


def _compact_combo(min_chars: int) -> QComboBox:
    """Combo box sized to a character count rather than its longest entry."""
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    return combo


__all__ = ["ExperimentView"]
