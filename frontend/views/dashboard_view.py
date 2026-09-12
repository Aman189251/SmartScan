"""Main dashboard: current scan, why it was chosen, and how the run is going."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..visualization.charts import BandBeliefChart, CurveChart, ScanTimelineChart
from ..visualization.spectrum import SpectrumTimeView
from ..widgets.cards import Card, PanelMenu, StatTile, ViewHeading
from ..widgets.decision_panel import DecisionPanel


class DashboardView(QWidget):
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
            "Live run", "The current scan, the decision behind it, and how the run is going."))
        toolbar.addStretch(1)
        toolbar.addWidget(self.panel_menu)
        layout.addLayout(toolbar)

        self.tiles = self._build_tiles()
        layout.addWidget(self.tiles)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        self.spectrum_card = Card(
            "Spectrum / time view",
            description="Where the receiver looked, drawn over the activity that was "
                        "actually there.")
        self.spectrum = SpectrumTimeView()
        self.spectrum_card.add(self.spectrum, 1)
        left_layout.addWidget(self.spectrum_card, 3)

        self.belief_card = Card(
            "Model probability against current belief",
            description="Prior knowledge next to live evidence, per band. The gap "
                        "between the two bars is what the scheduler is reasoning about.")
        self.band_chart = BandBeliefChart()
        self.belief_card.add(self.band_chart, 1)
        left_layout.addWidget(self.belief_card, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        self.decision_panel = DecisionPanel()
        right_layout.addWidget(self.decision_panel)

        self.timeline_card = Card(
            "Recent scan timeline",
            description="One dot per look: which band, which slot, and how it turned out.")
        self.timeline = ScanTimelineChart()
        self.timeline_card.add(self.timeline, 1)
        right_layout.addWidget(self.timeline_card, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        curves = QHBoxLayout()
        curves.setSpacing(14)

        self.rate_card = Card(
            "Rolling detection rate",
            description="Detections per scan over a moving window, as the run proceeds.")
        self.rate_curve = CurveChart("detections / scan")
        self.rate_card.add(self.rate_curve, 1)
        curves.addWidget(self.rate_card, 1)

        self.reward_card = Card(
            "Cumulative reward",
            description="Total reward banked since the run started. A steeper line is "
                        "a policy finding activity faster.")
        self.reward_curve = CurveChart("reward")
        self.reward_card.add(self.reward_curve, 1)
        curves.addWidget(self.reward_card, 1)
        layout.addLayout(curves)

        self.panel_menu.add_panel(self.tiles, "Summary tiles")
        self.panel_menu.add_panels([
            self.spectrum_card, self.belief_card, self.decision_panel,
            self.timeline_card, self.rate_card, self.reward_card,
        ])

    # ------------------------------------------------------------------ build
    def _build_tiles(self) -> QWidget:
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        self.tile_run = StatTile("RUN", "Idle", "no run in progress", small=True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximum(1000)
        self.tile_run.add(self.progress)

        self.tile_scan = StatTile("CURRENT SCAN", "--", "waiting", small=True)
        self.tile_intercept = StatTile("INTERCEPT RATE", "--", "detections per slot")
        self.tile_pd = StatTile("MEASURED Pd", "--", "detections when activity present")
        self.tile_pfa = StatTile("MEASURED Pfa", "--", "false alarms on quiet bands")
        self.tile_adaptive = StatTile("ADAPTIVE LAYER", "--", "background learner", small=True)

        tiles = [self.tile_run, self.tile_scan, self.tile_intercept,
                 self.tile_pd, self.tile_pfa, self.tile_adaptive]
        for column, tile in enumerate(tiles):
            grid.addWidget(tile, 0, column)
            grid.setColumnStretch(column, 1)
        return holder

    # ----------------------------------------------------------------- update
    def update_status(self, status: Dict[str, Any]) -> None:
        state = str(status.get("state", "IDLE"))
        slot = int(status.get("slot", 0) or 0)
        total = int(status.get("n_slots", 0) or 0)
        colors = {"RUNNING": theme.HIT, "PAUSED": theme.WARN, "FINISHED": theme.ACCENT_DEEP,
                  "STOPPED": theme.TEXT_DIM, "ERROR": theme.DANGER}
        self.tile_run.set_value(state.title(), colors.get(state, theme.TEXT_DIM))
        self.tile_run.set_caption(
            f"slot {slot:,} of {total:,}  -  {status.get('strategy', '--')}"
            if total else "no run in progress")
        self.progress.setValue(int(float(status.get("progress", 0.0)) * 1000))

        current = status.get("current")
        if current:
            outcome = str(current.get("outcome", ""))
            self.tile_scan.set_value(str(current.get("label", "--")),
                                     theme.OUTCOME_COLORS.get(outcome, theme.TEXT))
            self.tile_scan.set_caption(outcome.replace("_", " ").title())

        adaptive_state = str(status.get("adaptive_state", "--"))
        self.tile_adaptive.set_value(
            adaptive_state.replace("_", " ").title(),
            theme.ADAPTIVE_COLORS.get(adaptive_state, theme.TEXT_DIM))
        influence = float(status.get("adaptive_influence", 0.0) or 0.0)
        drift = " - drift flagged" if status.get("drift_flag") else ""
        self.tile_adaptive.set_caption(f"influence {influence:.0%}{drift}")

    def update_snapshot(self, snapshot: Dict[str, Any]) -> None:
        metrics = snapshot.get("metrics", {}) or {}
        self.tile_intercept.set_value(_fmt(metrics.get("average_intercept_rate")))
        self.tile_intercept.set_caption(
            f"{metrics.get('hits', 0)} detections in {metrics.get('scans', 0)} looks")
        self.tile_pd.set_value(_fmt(metrics.get("probability_of_detection")), theme.HIT)
        self.tile_pd.set_caption(
            f"{metrics.get('hits', 0)} of {int(metrics.get('hits', 0)) + int(metrics.get('misses', 0))} "
            f"looks at active bands")
        self.tile_pfa.set_value(_fmt(metrics.get("probability_of_false_alarm")), theme.FALSE_ALARM)
        self.tile_pfa.set_caption(f"{metrics.get('false_alarms', 0)} false alarms")

        status = {k: snapshot.get(k) for k in ("state", "current", "strategy", "slot")}
        since_scan = None
        current = snapshot.get("current")
        timeline = snapshot.get("timeline") or []
        if current and timeline:
            band = current.get("band")
            previous = [e for e in timeline[:-1] if e.get("band") == band]
            if previous:
                since_scan = int(current.get("slot", 0)) - int(previous[-1]["slot"])
        self.decision_panel.update_decision(status, snapshot.get("bands"), since_scan)

        ml = snapshot.get("ml_probabilities") or []
        belief = snapshot.get("posterior_means") or []
        selected = current.get("band") if current else None
        self.band_chart.update_bars(ml, belief, selected)

        n_bands = len(ml) or 32
        self.timeline.update_timeline(timeline, n_bands)

        curves = snapshot.get("curves", {}) or {}
        self.rate_curve.set_series(
            "rolling", curves.get("rolling_detection_rate", []), theme.HIT,
            label="Rolling detection rate",
            note="Detections per scan over a moving window.")
        self.reward_curve.set_series(
            "reward", curves.get("cumulative_reward", []), theme.ACCENT_DEEP,
            label="Cumulative reward",
            note="Running total of reward earned since the run started.")

    def update_activity(self, payload: Dict[str, Any], current_band: Optional[int]) -> None:
        self.spectrum.update_view(payload, current_band)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:            # NaN
        return "--"
    return f"{number:.{digits}f}"


__all__ = ["DashboardView"]
