"""Presentation primitives: cards, stat tiles, badges, key/value rows, switches.

``Card`` is the single container primitive.  Every card carries a title, an
optional one-line description of what the panel is for, and two controls: a
chevron that collapses the body away, and a button that lifts the body into a
resizable window for a proper look at a dense chart or a long table.

Collapsing is a layout change rather than a repaint trick - the body widget is
hidden and the card's vertical policy drops to ``Fixed``, so a collapsed card
shrinks to its header instead of holding empty space open.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme

#: Qt's "no maximum" sentinel; restoring it ends a collapse animation cleanly.
QWIDGETSIZE_MAX = 16_777_215


# --------------------------------------------------------------------- icons
def _icon(kind: str, color: str = theme.TEXT_DIM, size: int = 16) -> QIcon:
    """Small vector glyph drawn in code.

    Drawn rather than typed: the chevron and expand glyphs in the Unicode
    arrows block render inconsistently across the fonts installed on a given
    Windows machine, and a missing glyph shows as a replacement box.
    """
    scale = 2                                   # draw large, let Qt downsample
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(1.7 * scale)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    s = size * scale
    path = QPainterPath()

    if kind == "chevron-down":
        path.moveTo(s * 0.28, s * 0.40)
        path.lineTo(s * 0.50, s * 0.62)
        path.lineTo(s * 0.72, s * 0.40)
    elif kind == "chevron-right":
        path.moveTo(s * 0.40, s * 0.28)
        path.lineTo(s * 0.62, s * 0.50)
        path.lineTo(s * 0.40, s * 0.72)
    elif kind == "expand":                       # four outward corner brackets
        for x0, y0, dx, dy in ((0.22, 0.40, 0, -0.18), (0.78, 0.40, 0, -0.18),
                               (0.22, 0.60, 0, 0.18), (0.78, 0.60, 0, 0.18)):
            path.moveTo(s * x0, s * (y0 + dy))
            path.lineTo(s * (x0 + (0.18 if x0 < 0.5 else -0.18)), s * (y0 + dy))
            path.moveTo(s * x0, s * (y0 + dy))
            path.lineTo(s * x0, s * y0)
    elif kind == "collapse":                     # inward brackets
        for x0, y0, sx, sy in ((0.40, 0.40, 1, 1), (0.60, 0.40, -1, 1),
                               (0.40, 0.60, 1, -1), (0.60, 0.60, -1, -1)):
            path.moveTo(s * (x0 - 0.18 * sx), s * y0)
            path.lineTo(s * x0, s * y0)
            path.lineTo(s * x0, s * (y0 - 0.18 * sy))
    elif kind == "sliders":                      # panel-visibility menu glyph
        for y in (0.32, 0.50, 0.68):
            path.moveTo(s * 0.20, s * y)
            path.lineTo(s * 0.80, s * y)
        painter.drawPath(path)
        path = QPainterPath()
        painter.setBrush(QColor(color))
        for x, y in ((0.36, 0.32), (0.62, 0.50), (0.44, 0.68)):
            painter.drawEllipse(QRectF(s * (x - 0.07), s * (y - 0.07), s * 0.14, s * 0.14))

    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


class _ClickableLabel(QLabel):
    """Label that reports clicks, so a card title can toggle its own body."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:   # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ---------------------------------------------------------------------- card
class Card(QFrame):
    """Titled panel: the single container primitive across the app."""

    collapsed_changed = Signal(bool)

    def __init__(self, title: str = "", parent: Optional[QWidget] = None,
                 accent: bool = False, spacing: int = 10,
                 description: str = "", collapsible: bool = True,
                 expandable: bool = True, collapsed: bool = False):
        super().__init__(parent)
        self.setObjectName("CardAccent" if accent else "Card")
        self.title_text = title
        self._collapsed = False
        self._dialog: Optional[QDialog] = None
        self._animation: Optional[QPropertyAnimation] = None
        self._policy = self.sizePolicy()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(15, 12, 15, 14)
        outer.setSpacing(9)
        self._outer = outer

        # -- header -------------------------------------------------------
        self.header = QHBoxLayout()
        self.header.setSpacing(9)

        self._chevron = QPushButton()
        self._chevron.setObjectName("IconButton")
        self._chevron.setIcon(_icon("chevron-down"))
        self._chevron.setIconSize(QSize(15, 15))
        self._chevron.setCursor(Qt.PointingHandCursor)
        self._chevron.setToolTip("Collapse this panel")
        self._chevron.clicked.connect(self.toggle_collapsed)
        self._chevron.setVisible(collapsible)
        self.header.addWidget(self._chevron)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.title_label = _ClickableLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.title_label.setCursor(Qt.PointingHandCursor if collapsible else Qt.ArrowCursor)
        if collapsible:
            self.title_label.clicked.connect(self.toggle_collapsed)
        titles.addWidget(self.title_label)

        self.desc_label = QLabel(description)
        self.desc_label.setObjectName("CardDesc")
        self.desc_label.setWordWrap(True)
        self.desc_label.setVisible(bool(description))
        titles.addWidget(self.desc_label)
        self.header.addLayout(titles, 1)

        self._expand_button = QPushButton()
        self._expand_button.setObjectName("IconButton")
        self._expand_button.setIcon(_icon("expand"))
        self._expand_button.setIconSize(QSize(15, 15))
        self._expand_button.setCursor(Qt.PointingHandCursor)
        self._expand_button.setToolTip("Open this panel in a larger window")
        self._expand_button.clicked.connect(self.open_expanded)
        self._expand_button.setVisible(expandable)
        self.header.addWidget(self._expand_button)

        # Always installed, even for an untitled card: the header owns the
        # chevron and expand buttons, and a layout that is never installed
        # leaves its widgets parentless rather than merely hidden.
        outer.addLayout(self.header)

        # -- body ---------------------------------------------------------
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(spacing)
        outer.addWidget(self._content, 1)
        self._content_index = outer.count() - 1

        self._placeholder = QLabel("Showing in a separate window.")
        self._placeholder.setObjectName("Faint")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setVisible(False)
        outer.addWidget(self._placeholder)

        if collapsed and collapsible:
            self.set_collapsed(True, animate=False)

    # -------------------------------------------------------------- content
    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """Place a widget in the header, left of the expand button."""
        self.header.insertWidget(self.header.count() - 1, widget)
        return widget

    def set_title(self, title: str) -> None:
        self.title_text = title
        self.title_label.setText(title)

    def set_description(self, description: str) -> None:
        self.desc_label.setText(description)
        self.desc_label.setVisible(bool(description))

    def body(self) -> QVBoxLayout:
        return self._layout

    # ------------------------------------------------------------- collapse
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        if collapsed == self._collapsed or self._dialog is not None:
            return
        self._collapsed = collapsed
        self._chevron.setIcon(_icon("chevron-right" if collapsed else "chevron-down"))
        self._chevron.setToolTip("Expand this panel" if collapsed else "Collapse this panel")

        if collapsed:
            start = self._content.height()
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self._run_animation(start, 0, animate, self._after_collapse)
        else:
            self._content.setVisible(True)
            self.setSizePolicy(self._policy)
            target = max(self._content.sizeHint().height(), 1)
            self._run_animation(0, target, animate, self._after_expand)
        self.collapsed_changed.emit(collapsed)

    def _after_collapse(self) -> None:
        self._content.setVisible(False)
        self._content.setMaximumHeight(QWIDGETSIZE_MAX)

    def _after_expand(self) -> None:
        self._content.setMaximumHeight(QWIDGETSIZE_MAX)

    def _run_animation(self, start: int, end: int, animate: bool,
                       finished: Callable[[], None]) -> None:
        if self._animation is not None:
            self._animation.stop()
            self._animation = None
        if not animate:
            self._content.setMaximumHeight(end)
            finished()
            return
        animation = QPropertyAnimation(self._content, b"maximumHeight", self)
        animation.setDuration(150)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        animation.finished.connect(finished)
        self._animation = animation
        animation.start(QPropertyAnimation.DeleteWhenStopped)

    # --------------------------------------------------------------- expand
    def open_expanded(self) -> None:
        """Lift the body into its own resizable window.

        The body widget itself moves, rather than being duplicated, so whatever
        is driving it keeps its references and live updates continue to land in
        the expanded view.
        """
        if self._dialog is not None:
            self._dialog.raise_()
            self._dialog.activateWindow()
            return
        if self._collapsed:
            self.set_collapsed(False, animate=False)

        dialog = QDialog(self.window())
        dialog.setWindowTitle(self.title_text or "Panel")
        dialog.setStyleSheet(theme.STYLESHEET)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(10)

        heading = QLabel(self.title_text)
        heading.setObjectName("CardTitle")
        layout.addWidget(heading)
        if self.desc_label.text():
            caption = QLabel(self.desc_label.text())
            caption.setObjectName("CardDesc")
            caption.setWordWrap(True)
            layout.addWidget(caption)

        self._outer.removeWidget(self._content)
        self._content.setParent(dialog)
        self._content.setMaximumHeight(QWIDGETSIZE_MAX)
        self._content.setVisible(True)
        layout.addWidget(self._content, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_button = QPushButton("Close")
        close_button.setObjectName("Primary")
        close_button.clicked.connect(dialog.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        self._placeholder.setVisible(True)
        self._chevron.setEnabled(False)
        self._expand_button.setIcon(_icon("collapse"))
        self._expand_button.setToolTip("Bring this panel back into the page")

        parent_size = self.window().size()
        dialog.resize(int(parent_size.width() * 0.82), int(parent_size.height() * 0.82))
        dialog.finished.connect(self._close_expanded)
        self._dialog = dialog
        dialog.show()

    def _close_expanded(self) -> None:
        if self._dialog is None:
            return
        dialog = self._dialog
        self._dialog = None
        dialog.layout().removeWidget(self._content)
        self._content.setParent(self)
        self._outer.insertWidget(self._content_index, self._content, 1)
        self._content.setVisible(not self._collapsed)
        self._placeholder.setVisible(False)
        self._chevron.setEnabled(True)
        self._expand_button.setIcon(_icon("expand"))
        self._expand_button.setToolTip("Open this panel in a larger window")
        dialog.deleteLater()


# ----------------------------------------------------------------- stat tile
class StatTile(Card):
    """One headline number with a caption and optional coloured value."""

    def __init__(self, title: str, value: str = "--", caption: str = "",
                 parent: Optional[QWidget] = None, small: bool = False):
        super().__init__(title, parent, collapsible=False, expandable=False)
        self.title_label.setObjectName("StatCaption")
        self.title_label.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600;"
            "letter-spacing: 0.4px;")
        self._outer.setContentsMargins(14, 11, 14, 12)
        self._outer.setSpacing(4)
        self.body().setSpacing(2)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatValueSmall" if small else "StatValue")
        self.value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.add(self.value_label)

        self.caption_label = QLabel(caption)
        self.caption_label.setObjectName("StatCaption")
        self.caption_label.setWordWrap(True)
        self.add(self.caption_label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: str, color: Optional[str] = None) -> None:
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color};" if color else "")

    def set_caption(self, caption: str) -> None:
        self.caption_label.setText(caption)


class Badge(QLabel):
    """Small coloured status pill."""

    def __init__(self, text: str = "", color: str = theme.TEXT_DIM,
                 parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        # A pill, never a panel: left to the default policy a QLabel grows to
        # fill whatever vertical space its row is given, and the rounded
        # background stretches into a block.
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(theme.badge_style(color))

    def set_state(self, text: str, color: str) -> None:
        self.setText(text)
        self.set_color(color)


class KeyValueGrid(QWidget):
    """Two-column label/value grid with monospaced values."""

    def __init__(self, rows: Iterable[Tuple[str, str]] = (), parent: Optional[QWidget] = None,
                 columns: int = 1):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(6)
        self._columns = max(1, columns)
        self._values = {}
        self._count = 0
        for key, value in rows:
            self.add_row(key, value)

    def add_row(self, key: str, value: str = "--") -> QLabel:
        column_block = self._count % self._columns
        row = self._count // self._columns
        key_label = QLabel(key)
        key_label.setObjectName("Dim")
        value_label = QLabel(value)
        value_label.setObjectName("Mono")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._grid.addWidget(key_label, row, column_block * 2)
        self._grid.addWidget(value_label, row, column_block * 2 + 1)
        self._grid.setColumnStretch(column_block * 2 + 1, 1)
        self._values[key] = value_label
        self._count += 1
        return value_label

    def set(self, key: str, value: str, color: Optional[str] = None) -> None:
        label = self._values.get(key)
        if label is None:
            label = self.add_row(key, value)
        label.setText(value)
        label.setStyleSheet(f"color: {color};" if color else "")

    def keys(self):
        return self._values.keys()


class SectionLabel(QLabel):
    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setObjectName("CardTitle")


class ViewHeading(QWidget):
    """Title and one-line purpose, shown at the top of a view."""

    def __init__(self, title: str, description: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(1)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {theme.TEXT}; font-size: 17px; font-weight: 700;")
        layout.addWidget(self.title_label)

        self.desc_label = QLabel(description)
        self.desc_label.setObjectName("CardDesc")
        self.desc_label.setWordWrap(True)
        self.desc_label.setVisible(bool(description))
        layout.addWidget(self.desc_label)


# ------------------------------------------------------------- panel control
class PanelMenu(QPushButton):
    """Button opening a checklist of the panels on a view.

    Answers "show me only what I care about": every registered card can be
    switched off entirely, and the whole view collapsed or expanded at once.
    """

    def __init__(self, text: str = "Panels", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setIcon(_icon("sliders"))
        self.setIconSize(QSize(15, 15))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Choose which panels this view shows")
        self._menu = QMenu(self)
        self._cards: List[Tuple[QWidget, object]] = []

        self._menu.addAction("Show every panel", self.show_all)
        self._menu.addAction("Expand every panel", lambda: self._set_all_collapsed(False))
        self._menu.addAction("Collapse every panel", lambda: self._set_all_collapsed(True))
        self._menu.addSeparator()
        self.setMenu(self._menu)

    def add_panel(self, panel: QWidget, name: str = "") -> None:
        """Register a panel. Anything with a title works; a plain widget needs
        *name* because only a :class:`Card` can supply one itself."""
        label = name or getattr(panel, "title_text", "") or panel.objectName() or "Panel"
        action = self._menu.addAction(label)
        action.setCheckable(True)
        # isVisible() is False for every widget here, because panels are
        # registered while the view is still being built and nothing has been
        # shown yet.  isHidden() asks the question actually meant: has this
        # panel been switched off explicitly?
        action.setChecked(not panel.isHidden())
        action.toggled.connect(panel.setVisible)
        self._cards.append((panel, action))

    def add_panels(self, panels: Iterable[QWidget]) -> None:
        for panel in panels:
            self.add_panel(panel)

    def show_all(self) -> None:
        for panel, action in self._cards:
            action.setChecked(True)
            panel.setVisible(True)

    def _set_all_collapsed(self, collapsed: bool) -> None:
        for panel, _ in self._cards:
            if isinstance(panel, Card):
                panel.set_collapsed(collapsed, animate=False)


class ToggleSwitch(QAbstractButton):
    """iOS-style switch: a checkbox with the affordance the mock-up uses."""

    def __init__(self, parent: Optional[QWidget] = None, checked: bool = False):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(40, 22)

    def sizeHint(self) -> QSize:                # noqa: N802 - Qt naming
        return QSize(40, 22)

    def paintEvent(self, event) -> None:        # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = QRectF(0.5, 2.5, self.width() - 1, self.height() - 5)
        on = self.isChecked()
        enabled = self.isEnabled()

        fill = QColor(theme.ACCENT if on else theme.BORDER_LIGHT)
        if not enabled:
            fill.setAlpha(90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        diameter = track.height() - 4
        x = track.right() - diameter - 2 if on else track.left() + 2
        knob = QColor("#ffffff")
        if not enabled:
            knob.setAlpha(190)
        painter.setBrush(knob)
        painter.drawEllipse(QRectF(x, track.top() + 2, diameter, diameter))
        painter.end()


__all__ = [
    "Card", "StatTile", "Badge", "KeyValueGrid", "SectionLabel", "ViewHeading",
    "PanelMenu", "ToggleSwitch",
]
