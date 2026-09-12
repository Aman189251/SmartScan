"""Live technical charts: per-band belief bars, scan timeline, metric curves.

Every chart here pairs a pyqtgraph plot with a :class:`ChartLegend` underneath,
and each class keeps its legend in step with what it has actually drawn.  A
series that is not on the plot does not appear in the legend, and a colour on
the plot always appears in the legend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import theme
from ..widgets.chart_legend import ChartLegend


def _style_plot(plot: pg.PlotWidget, x_label: str = "", y_label: str = "",
                grid_x: bool = False, grid_y: bool = True) -> pg.PlotWidget:
    """Apply the light chart language: no fill, hairline axes, soft grid."""
    plot.setBackground(None)                     # the card's glass shows through
    plot.setMenuEnabled(False)
    plot.hideButtons()
    for axis in ("left", "bottom"):
        plot.getAxis(axis).setPen(pg.mkPen(theme.AXIS))
        plot.getAxis(axis).setTextPen(pg.mkPen(theme.TEXT_DIM))
    if x_label:
        plot.setLabel("bottom", x_label, color=theme.TEXT_DIM)
    if y_label:
        plot.setLabel("left", y_label, color=theme.TEXT_DIM)
    plot.showGrid(x=grid_x, y=grid_y, alpha=theme.GRID_ALPHA)
    return plot


class _ChartBase(QWidget):
    """A plot with a legend strip beneath it."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(7)
        self.legend = ChartLegend()

    def _finish(self, plot: pg.PlotWidget, stretch: int = 1) -> None:
        self._root.addWidget(plot, stretch)
        self._root.addWidget(self.legend)


class BandBeliefChart(_ChartBase):
    """Model probability against current belief for every band.

    Two overlaid bar series make the core idea visible at a glance: what the
    offline model expects, and what live evidence currently supports.
    """

    def __init__(self, parent: Optional[QWidget] = None, height: int = 175):
        super().__init__(parent)
        self.plot = _style_plot(pg.PlotWidget(), "Abstract band", "Probability")
        self.plot.setMinimumHeight(height)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(0.0, 1.0, padding=0.02)

        self.ml_bars = pg.BarGraphItem(x=[], height=[], width=0.38,
                                       brush=theme.ACCENT_DEEP, pen=pg.mkPen(None))
        self.belief_bars = pg.BarGraphItem(x=[], height=[], width=0.38,
                                           brush=theme.HIT, pen=pg.mkPen(None))
        self.plot.addItem(self.ml_bars)
        self.plot.addItem(self.belief_bars)

        self.selected = pg.ScatterPlotItem(pen=pg.mkPen(theme.PRIMARY, width=2),
                                           brush=pg.mkBrush(None), size=14, symbol="t1")
        self.plot.addItem(self.selected)
        self._finish(self.plot)

        self.legend.set_entries([
            ("Model probability", theme.ACCENT_DEEP, "bar",
             "What the offline model expects for this band, before live evidence."),
            ("Bayesian belief", theme.HIT, "bar",
             "What the looks taken so far actually support."),
            ("Band being scanned now", theme.PRIMARY, "triangle",
             "The band the scheduler selected for the current slot."),
        ])

    def update_bars(self, ml: Sequence[float], belief: Sequence[float],
                    selected_band: Optional[int] = None) -> None:
        if not ml:
            return
        x = np.arange(len(ml), dtype=float)
        self.ml_bars.setOpts(x=x - 0.20, height=np.asarray(ml, dtype=float), width=0.38)
        if belief:
            self.belief_bars.setOpts(x=x[:len(belief)] + 0.20,
                                     height=np.asarray(belief, dtype=float), width=0.38)
        if selected_band is not None and 0 <= selected_band < len(ml):
            top = max(float(ml[selected_band]),
                      float(belief[selected_band]) if belief else 0.0)
            self.selected.setData([selected_band], [min(1.0, top + 0.06)])
        else:
            self.selected.setData([], [])


class ScanTimelineChart(_ChartBase):
    """Recent scan decisions plotted as band against slot, coloured by outcome."""

    #: Plain-language names for the four outcome classes.
    OUTCOME_LABELS = {
        "HIT": ("Detection", "Looked at an active band and saw it."),
        "MISS": ("Missed activity", "Looked at an active band and saw nothing."),
        "FALSE_ALARM": ("False alarm", "Reported energy on a band that was quiet."),
        "CORRECT_NON_DETECTION": ("Correctly quiet", "Looked at a quiet band and saw nothing."),
    }

    def __init__(self, parent: Optional[QWidget] = None, height: int = 160):
        super().__init__(parent)
        self.plot = _style_plot(pg.PlotWidget(), "Time slot", "Band", grid_y=True)
        self.plot.setMinimumHeight(height)
        self.plot.setMouseEnabled(x=False, y=False)

        self._series: Dict[str, pg.ScatterPlotItem] = {}
        entries = []
        for outcome, color in theme.OUTCOME_COLORS.items():
            hollow = outcome == "CORRECT_NON_DETECTION"
            scatter = pg.ScatterPlotItem(
                size=8, symbol="o",
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(None) if hollow else pg.mkBrush(color))
            self.plot.addItem(scatter)
            self._series[outcome] = scatter
            label, note = self.OUTCOME_LABELS[outcome]
            entries.append((label, color, "hollow" if hollow else "dot", note))

        self._finish(self.plot)
        self.legend.set_entries(entries)

    def update_timeline(self, entries: List[Dict[str, Any]], n_bands: int = 32) -> None:
        buckets: Dict[str, List[List[float]]] = {k: [[], []] for k in self._series}
        for entry in entries:
            outcome = entry.get("outcome", "CORRECT_NON_DETECTION")
            if outcome not in buckets:
                continue
            buckets[outcome][0].append(entry["slot"])
            buckets[outcome][1].append(entry["band"])
        for outcome, scatter in self._series.items():
            scatter.setData(buckets[outcome][0], buckets[outcome][1])
        if entries:
            self.plot.setXRange(entries[0]["slot"], entries[-1]["slot"] + 1, padding=0.02)
        self.plot.setYRange(-0.5, n_bands - 0.5, padding=0.02)


class CurveChart(_ChartBase):
    """Single- or multi-series line chart for metric curves.

    Series register themselves in the legend as they are first drawn, so a
    caller only has to name a line once and it is explained from then on.
    """

    def __init__(self, y_label: str = "", parent: Optional[QWidget] = None,
                 height: int = 145, x_label: str = "Scan"):
        super().__init__(parent)
        self.plot = _style_plot(pg.PlotWidget(), x_label, y_label)
        self.plot.setMinimumHeight(height)
        self.plot.setMouseEnabled(x=False, y=False)
        self._finish(self.plot)

        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._markers: List[pg.InfiniteLine] = []
        self._entries: List[tuple] = []
        self._marker_entry: Optional[tuple] = None

    def set_series(self, name: str, values: Sequence[float], color: str = theme.ACCENT_DEEP,
                   width: int = 2, label: str = "", note: str = "") -> None:
        if name not in self._curves:
            self._curves[name] = self.plot.plot(
                [], [], pen=pg.mkPen(color, width=width), name=name)
            self._entries.append((label or name, color, "line", note))
            self._refresh_legend()
        self._curves[name].setData(np.arange(len(values)), np.asarray(values, dtype=float))

    def mark(self, positions: Sequence[float], color: str = theme.WARN,
             label: str = "Marked event") -> None:
        for line in self._markers:
            self.plot.removeItem(line)
        self._markers.clear()
        for pos in positions:
            line = pg.InfiniteLine(pos=pos, angle=90,
                                   pen=pg.mkPen(color, width=1, style=Qt.DashLine))
            self.plot.addItem(line)
            self._markers.append(line)
        entry = (label, color, "dashed", "") if positions else None
        if entry != self._marker_entry:
            self._marker_entry = entry
            self._refresh_legend()

    def _refresh_legend(self) -> None:
        entries = list(self._entries)
        if self._marker_entry is not None:
            entries.append(self._marker_entry)
        self.legend.set_entries(entries)

    def clear(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])


class ComparisonChart(_ChartBase):
    """Grouped bar chart comparing strategies on one metric."""

    def __init__(self, parent: Optional[QWidget] = None, height: int = 260):
        super().__init__(parent)
        self.plot = _style_plot(pg.PlotWidget(), "", "")
        self.plot.setMinimumHeight(height)
        self.plot.setMouseEnabled(x=False, y=False)
        self._finish(self.plot)

        self.bars: Optional[pg.BarGraphItem] = None
        self._error: Optional[pg.ErrorBarItem] = None
        self.legend.set_entries([
            ("Best on this metric", theme.PRIMARY, "bar",
             "The winning strategy, whether that means the largest or the smallest value."),
            ("Other strategies", theme.ACCENT_DEEP, "bar", ""),
            ("±1 standard deviation across seeds", theme.TEXT_FAINT, "whisker",
             "Spread of the per-seed results behind each mean."),
        ])

    def update_bars(self, labels: Sequence[str], values: Sequence[float],
                    errors: Optional[Sequence[float]] = None, metric: str = "",
                    lower_is_better: bool = False) -> None:
        """Draw one bar per strategy, highlighting the best.

        ``lower_is_better`` matters: for intercept delay and false-alarm rate the
        winner is the smallest bar, and colouring the tallest one as the winner
        would say the opposite of the truth.
        """
        self.plot.clear()
        if not labels:
            return
        x = np.arange(len(labels), dtype=float)
        heights = np.asarray(values, dtype=float)
        if len(heights) == 0:
            best = -1
        else:
            best = int(np.argmin(heights) if lower_is_better else np.argmax(heights))
        brushes = [pg.mkBrush(theme.PRIMARY if i == best else theme.ACCENT_DEEP)
                   for i in range(len(heights))]
        self.bars = pg.BarGraphItem(x=x, height=heights, width=0.6, brushes=brushes,
                                    pen=pg.mkPen(None))
        self.plot.addItem(self.bars)
        if errors is not None and len(errors) == len(heights):
            self._error = pg.ErrorBarItem(x=x, y=heights, height=np.asarray(errors) * 2,
                                          beam=0.16, pen=pg.mkPen(theme.TEXT_FAINT, width=1))
            self.plot.addItem(self._error)
        axis = self.plot.getAxis("bottom")
        axis.setTicks([[(i, label) for i, label in enumerate(labels)]])
        self.plot.setLabel("left", metric, color=theme.TEXT_DIM)

        direction = "lowest wins" if lower_is_better else "highest wins"
        self.legend.set_entries([
            (f"Best on this metric ({direction})", theme.PRIMARY, "bar", ""),
            ("Other strategies", theme.ACCENT_DEEP, "bar", ""),
            ("±1 standard deviation across seeds", theme.TEXT_FAINT, "whisker",
             "Spread of the per-seed results behind each mean."),
        ])


class ImportanceChart(_ChartBase):
    """Horizontal bars for model feature importance."""

    def __init__(self, parent: Optional[QWidget] = None, height: int = 210):
        super().__init__(parent)
        self.plot = _style_plot(pg.PlotWidget(), "Weight", "", grid_x=True, grid_y=False)
        self.plot.setMinimumHeight(height)
        self.plot.setMouseEnabled(x=False, y=False)
        self._finish(self.plot)
        self.legend.set_entries([
            ("Relative contribution to the model", theme.ACCENT_DEEP, "bar",
             "How much the trained model leans on each feature. Observation "
             "history only - ground truth is never a feature."),
        ])

    def update_importance(self, importance: Dict[str, float], top: int = 12) -> None:
        self.plot.clear()
        if not importance:
            return
        ordered = sorted(importance.items(), key=lambda kv: float(kv[1]), reverse=True)[:top]
        ordered.reverse()                        # largest at the top of the axis
        names = [name for name, _ in ordered]
        weights = np.asarray([float(value) for _, value in ordered], dtype=float)
        y = np.arange(len(names), dtype=float)

        bars = pg.BarGraphItem(x0=0, y=y, height=0.62, width=weights,
                               brush=theme.ACCENT_DEEP, pen=pg.mkPen(None))
        self.plot.addItem(bars)
        self.plot.getAxis("left").setTicks([[(i, name) for i, name in enumerate(names)]])
        self.plot.setYRange(-0.7, len(names) - 0.3, padding=0.02)


__all__ = ["BandBeliefChart", "ScanTimelineChart", "CurveChart", "ComparisonChart",
           "ImportanceChart"]
