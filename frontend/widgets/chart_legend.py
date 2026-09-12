"""Chart legends.

Every chart in the workstation carries one of these underneath it, so a colour
never appears on screen without a name attached to it.

The legend lives outside the plot rather than inside it.  pyqtgraph's built-in
legend is drawn as a plot item, which means ``PlotItem.clear()`` destroys it -
the comparison chart rebuilds itself on every metric change - and it floats over
the data at exactly the corner where a tall bar or a rising curve wants to be.
A widget below the axis has neither problem, and it can wrap onto a second line
when a card is narrow.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLayout, QSizePolicy, QWidget

from .. import theme

#: ``(label, colour, kind)`` - the shape drawn in the swatch.
LegendEntry = Tuple[str, str, str]

SWATCH_W = 22
SWATCH_H = 12


def _swatch(color: str, kind: str = "line") -> QPixmap:
    """Draw the mark this series actually uses on the plot."""
    scale = 2
    pixmap = QPixmap(SWATCH_W * scale, SWATCH_H * scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    w, h = SWATCH_W * scale, SWATCH_H * scale
    qcolor = QColor(color)

    if kind in ("line", "dashed"):
        pen = QPen(qcolor)
        pen.setWidthF(2.2 * scale)
        pen.setCapStyle(Qt.RoundCap)
        if kind == "dashed":
            pen.setStyle(Qt.DashLine)
            pen.setDashPattern([2.6, 2.0])
        painter.setPen(pen)
        painter.drawLine(int(w * 0.06), h // 2, int(w * 0.94), h // 2)
    elif kind == "bar":
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(qcolor))
        painter.drawRoundedRect(QRectF(w * 0.12, h * 0.14, w * 0.76, h * 0.72),
                                2.0 * scale, 2.0 * scale)
    elif kind == "heat":
        gradient_steps = 24
        painter.setPen(Qt.NoPen)
        for step in range(gradient_steps):
            fraction = step / (gradient_steps - 1)
            blended = QColor(color)
            blended.setAlphaF(0.10 + 0.90 * fraction)
            painter.setBrush(QBrush(blended))
            painter.drawRect(QRectF(w * 0.10 + (w * 0.80) * fraction / 1.0,
                                    h * 0.16, w * 0.80 / gradient_steps + 1, h * 0.68))
    elif kind == "dot":
        painter.setPen(QPen(qcolor, 1.2 * scale))
        painter.setBrush(QBrush(qcolor))
        painter.drawEllipse(QRectF(w * 0.34, h * 0.18, h * 0.64, h * 0.64))
    elif kind == "hollow":
        painter.setPen(QPen(qcolor, 1.6 * scale))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(w * 0.34, h * 0.18, h * 0.64, h * 0.64))
    elif kind == "square":
        painter.setPen(QPen(qcolor, 1.2 * scale))
        painter.setBrush(QBrush(qcolor))
        painter.drawRect(QRectF(w * 0.32, h * 0.18, h * 0.64, h * 0.64))
    elif kind == "square-hollow":
        painter.setPen(QPen(qcolor, 1.6 * scale))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(w * 0.32, h * 0.18, h * 0.64, h * 0.64))
    elif kind == "marker":                       # the ring used for "current"
        painter.setPen(QPen(qcolor, 1.8 * scale))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(w * 0.28, h * 0.10, h * 0.80, h * 0.80))
    elif kind == "triangle":                     # the pointer above a selected bar
        painter.setPen(QPen(qcolor, 1.8 * scale))
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(w * 0.50, h * 0.14)
        path.lineTo(w * 0.68, h * 0.84)
        path.lineTo(w * 0.32, h * 0.84)
        path.closeSubpath()
        painter.drawPath(path)
    elif kind == "whisker":
        pen = QPen(qcolor)
        pen.setWidthF(1.6 * scale)
        painter.setPen(pen)
        cx = w // 2
        painter.drawLine(cx, int(h * 0.12), cx, int(h * 0.88))
        painter.drawLine(int(w * 0.28), int(h * 0.12), int(w * 0.72), int(h * 0.12))
        painter.drawLine(int(w * 0.28), int(h * 0.88), int(w * 0.72), int(h * 0.88))

    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


class _FlowLayout(QLayout):
    """Left-to-right layout that wraps onto the next line when it runs out."""

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 14):
        super().__init__(parent)
        self._items: List = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:            # noqa: N802 - Qt naming
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):                    # noqa: N802 - Qt naming
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):                    # noqa: N802 - Qt naming
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):              # noqa: N802 - Qt naming
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:        # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self._layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:        # noqa: N802 - Qt naming
        super().setGeometry(rect)
        self._layout(rect, test_only=False)

    def sizeHint(self) -> QSize:                # noqa: N802 - Qt naming
        return self.minimumSize()

    def minimumSize(self) -> QSize:             # noqa: N802 - Qt naming
        # Width: the widest single entry, so no swatch/label pair is ever cut
        # in half.  Height: one row, because the real height is whatever
        # heightForWidth() works out once the column width is known.
        width, height = 0, 0
        for item in self._items:
            hint = item.sizeHint()
            width = max(width, hint.width())
            height = max(height, hint.height())
        return QSize(width, height)

    def _layout(self, rect: QRect, test_only: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._space
            if next_x - self._space > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + 4
                next_x = x + hint.width() + self._space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class _LegendItem(QWidget):
    """One swatch and its name."""

    def __init__(self, label: str, color: str, kind: str,
                 note: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        icon = QLabel()
        icon.setPixmap(_swatch(color, kind))
        icon.setFixedSize(SWATCH_W, SWATCH_H + 2)
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        text = QLabel(label)
        text.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12px;")
        layout.addWidget(text)
        if note:
            self.setToolTip(note)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class ChartLegend(QWidget):
    """Wrapping strip of swatch/label pairs shown beneath a chart."""

    def __init__(self, entries: Sequence[LegendEntry] = (),
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._flow = _FlowLayout(self, spacing=14)
        self._entries: List[LegendEntry] = []
        # A size policy only consults heightForWidth() when it is asked to;
        # without this the widget reserves the full stacked height of every
        # entry even when they all fit on one line.
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        if entries:
            self.set_entries(entries)

    def hasHeightForWidth(self) -> bool:        # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self._flow.heightForWidth(width)

    def set_entries(self, entries: Iterable[LegendEntry]) -> None:
        """Replace the legend contents.

        Each entry is ``(label, colour, kind)`` and may carry a fourth element
        used as the tooltip when a mark needs a sentence rather than a name.
        """
        while self._flow.count():
            item = self._flow.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._entries = []
        for entry in entries:
            label, color, kind = entry[0], entry[1], entry[2]
            note = entry[3] if len(entry) > 3 else ""
            self._entries.append((label, color, kind))
            self._flow.addWidget(_LegendItem(label, color, kind, note))
        self.updateGeometry()

    def entries(self) -> List[LegendEntry]:
        return list(self._entries)


__all__ = ["ChartLegend", "LegendEntry"]
