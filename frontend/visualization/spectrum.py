"""Band-by-time visualisation.

Shows the simulated activity grid, where the receiver actually looked, and how
those looks turned out.  This is a research and demonstration view of a
synthetic environment: the grid it draws is the simulator's own state, shown to
the operator after the fact, and it is never fed back into a decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import theme
from ..widgets.chart_legend import ChartLegend

pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")


def _colormap() -> pg.ColorMap:
    """Clear background to lit activity."""
    positions = [0.0, 1.0]
    colors = [pg.mkColor(255, 255, 255, 0), pg.mkColor(theme.ACTIVE_TRUTH)]
    return pg.ColorMap(positions, colors)


class SpectrumTimeView(QWidget):
    """Heatmap of hidden activity with the scan trail drawn over it."""

    def __init__(self, parent: Optional[QWidget] = None, height: int = 260):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self.plot = pg.PlotWidget(background=None)
        self.plot.setMinimumHeight(height)
        self.plot.setLabel("left", "Abstract band", color=theme.TEXT_DIM)
        self.plot.setLabel("bottom", "Time slot", color=theme.TEXT_DIM)
        self.plot.showGrid(x=False, y=False)
        self.plot.getAxis("left").setPen(pg.mkPen(theme.AXIS))
        self.plot.getAxis("bottom").setPen(pg.mkPen(theme.AXIS))
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme.TEXT_DIM))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(theme.TEXT_DIM))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        layout.addWidget(self.plot, 1)

        self.image = pg.ImageItem()
        self.image.setColorMap(_colormap())
        self.plot.addItem(self.image)

        # Scan overlay: one scatter per outcome class so colours stay meaningful.
        self._scatters: Dict[str, pg.ScatterPlotItem] = {}
        for key, color, symbol, size, filled in (
            ("hit", theme.HIT, "s", 7, True),
            ("quiet", theme.TEXT_FAINT, "s", 5, False),
        ):
            scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(color) if filled else pg.mkBrush(None),
                size=size, symbol=symbol)
            self.plot.addItem(scatter)
            self._scatters[key] = scatter

        self.current_marker = pg.ScatterPlotItem(
            pen=pg.mkPen(theme.PRIMARY, width=2), brush=pg.mkBrush(None), size=16, symbol="o"
        )
        self.plot.addItem(self.current_marker)
        self._change_lines: List[pg.InfiniteLine] = []
        self._n_bands = 0

        self.legend = ChartLegend([
            ("Simulated activity", theme.ACTIVE_TRUTH, "heat",
             "The simulator's own state, shown after the fact. It is never fed "
             "back into a decision."),
            ("Look that detected energy", theme.HIT, "square", ""),
            ("Look that found nothing", theme.TEXT_FAINT, "square-hollow", ""),
            ("Band being scanned now", theme.PRIMARY, "marker", ""),
            ("Environment shift", theme.WARN, "dashed",
             "The point where the scenario changed its activity pattern."),
        ])
        layout.addWidget(self.legend)

    def update_view(self, payload: Dict[str, Any], current_band: Optional[int] = None) -> None:
        grid = payload.get("grid") or []
        if not grid:
            return
        array = np.asarray(grid, dtype=float)          # (bands, slots)
        self._n_bands = array.shape[0]
        start = int(payload.get("start_slot", 0))
        width = array.shape[1]

        self.image.setImage(array, autoLevels=False, levels=(0.0, 1.0))
        self.image.setRect(pg.QtCore.QRectF(start, -0.5, max(1, width), self._n_bands))

        hits_x, hits_y, quiet_x, quiet_y = [], [], [], []
        for offset, band, detected in payload.get("scans", []):
            slot = start + offset
            if detected:
                hits_x.append(slot)
                hits_y.append(band)
            else:
                quiet_x.append(slot)
                quiet_y.append(band)
        self._scatters["hit"].setData(hits_x, hits_y)
        self._scatters["quiet"].setData(quiet_x, quiet_y)

        if current_band is not None and hits_x + quiet_x:
            self.current_marker.setData([start + width - 1], [current_band])
        else:
            self.current_marker.setData([], [])

        for line in self._change_lines:
            self.plot.removeItem(line)
        self._change_lines.clear()
        for offset in payload.get("change_points", []):
            line = pg.InfiniteLine(
                pos=start + offset, angle=90,
                pen=pg.mkPen(theme.WARN, width=2, style=Qt.DashLine),
                label="environment shift",
                labelOpts={"color": theme.WARN, "position": 0.92, "movable": False},
            )
            self.plot.addItem(line)
            self._change_lines.append(line)

        self.plot.setXRange(start, start + width, padding=0.01)
        self.plot.setYRange(-0.5, self._n_bands - 0.5, padding=0.01)

    def clear(self) -> None:
        self.image.clear()
        for scatter in self._scatters.values():
            scatter.setData([], [])
        self.current_marker.setData([], [])


__all__ = ["SpectrumTimeView"]
