"""Visual language for the workstation.

A light, glassy instrument panel: translucent white surfaces floating over a
soft page wash.  Two accents carry meaning and never swap roles - blue is
*information* (the model, the current scan, a selected series) and orange is
*action* (start, promote, the view you are standing in).  Outcome colours are
used identically everywhere, so an operator learns the palette once.

Surfaces are deliberately semi-opaque.  Cards sit at ~78% white over the page
gradient and charts paint on no background at all, so a panel reads as a sheet
of glass laid on the workspace rather than a box cut out of it.

No ``QGraphicsDropShadowEffect`` is used anywhere.  A graphics effect forces a
full re-composite of the widget beneath it on every repaint, and these cards
contain pyqtgraph views that redraw several times a second; elevation is
carried by the border and the translucency instead.
"""

from __future__ import annotations

# -- page ----------------------------------------------------------------
BG = "#eef2f7"
BG_TOP = "#f8fafc"
BG_BOTTOM = "#e7edf5"

# -- surfaces ------------------------------------------------------------
SURFACE = "#ffffff"
PANEL = "#ffffff"                 # kept: charts and tables reference it
PANEL_ALT = "#f1f5f9"
SIDEBAR = "#f7f9fc"
BORDER = "#e3e8ef"
BORDER_LIGHT = "#cbd5e1"

#: Card fill.  Semi-opaque so the page wash shows through.
GLASS = "rgba(255, 255, 255, 0.78)"
GLASS_STRONG = "rgba(255, 255, 255, 0.92)"
GLASS_HEADER = "rgba(255, 255, 255, 0.86)"

# -- ink -----------------------------------------------------------------
TEXT = "#1e293b"
TEXT_DIM = "#64748b"
TEXT_FAINT = "#94a3b8"

# -- accents -------------------------------------------------------------
ACCENT = "#3b82f6"                # information / model / selection
ACCENT_DEEP = "#2563eb"
ACCENT_SOFT = "#dbeafe"

PRIMARY = "#f97316"               # action / current view
PRIMARY_DEEP = "#ea580c"
PRIMARY_SOFT = "#ffedd5"

# -- semantic ------------------------------------------------------------
HIT = "#0d9488"
MISS = "#e11d48"
FALSE_ALARM = "#d97706"
QUIET = "#b8c4d4"
ACTIVE_TRUTH = "#c7dcfa"          # simulated activity shading on the heatmap
WARN = "#ea580c"
DANGER = "#dc2626"

OUTCOME_COLORS = {
    "HIT": HIT,
    "MISS": MISS,
    "FALSE_ALARM": FALSE_ALARM,
    "CORRECT_NON_DETECTION": QUIET,
}

ADAPTIVE_COLORS = {
    "OFFLINE_PRIOR": TEXT_FAINT,
    "LEARNING": ACCENT,
    "SHADOW": "#7c3aed",
    "ADAPTIVE_ASSISTED": PRIMARY,
    "ADAPTIVE_CONTROL": HIT,
}

#: Categorical series colours, in order, for charts carrying several lines.
SERIES = [ACCENT_DEEP, PRIMARY, HIT, "#7c3aed", MISS, "#0891b2"]

# -- charts --------------------------------------------------------------
AXIS = "#cbd5e1"
GRID_ALPHA = 0.28

MONO = "Consolas, 'Cascadia Mono', 'Courier New', monospace"
SANS = "'Segoe UI', 'Inter', system-ui, sans-serif"


STYLESHEET = f"""
QWidget {{
    background: transparent;
    color: {TEXT};
    font-family: {SANS};
    font-size: 13px;
}}

QMainWindow, QDialog {{ background: {BG}; }}

QWidget#Page {{
    background: qlineargradient(x1:0, y1:0, x2:0.35, y2:1,
        stop:0 {BG_TOP}, stop:0.5 {BG}, stop:1 {BG_BOTTOM});
}}

/* ---------- cards ---------- */
QFrame#Card {{
    background: {GLASS};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}
QFrame#CardAccent {{
    background: {GLASS_STRONG};
    border: 1px solid {ACCENT};
    border-radius: 14px;
}}
QFrame#Card:hover, QFrame#CardAccent:hover {{ border-color: {BORDER_LIGHT}; }}

QLabel#CardTitle {{
    color: {TEXT};
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.1px;
}}
QLabel#CardDesc {{
    color: {TEXT_DIM};
    font-size: 12px;
}}
QLabel#StatValue {{
    color: {TEXT};
    font-size: 25px;
    font-weight: 650;
    font-family: {MONO};
}}
QLabel#StatValueSmall {{
    color: {TEXT};
    font-size: 17px;
    font-weight: 650;
    font-family: {MONO};
}}
QLabel#StatCaption {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#Mono {{ font-family: {MONO}; color: {TEXT}; }}
QLabel#Dim {{ color: {TEXT_DIM}; }}
QLabel#Faint {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel#SectionCaption {{ color: {TEXT_DIM}; font-size: 12px; }}

/* ---------- card header buttons ---------- */
QPushButton#IconButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 0px;
    min-width: 26px;
    max-width: 26px;
    min-height: 24px;
    max-height: 24px;
    color: {TEXT_DIM};
    font-size: 13px;
}}
QPushButton#IconButton:hover {{
    background: {ACCENT_SOFT};
    border-color: {ACCENT};
    color: {ACCENT_DEEP};
}}
QPushButton#IconButton:pressed {{ background: {ACCENT_SOFT}; }}

/* ---------- header ---------- */
QFrame#Header {{
    background: {GLASS_HEADER};
    border-bottom: 1px solid {BORDER};
}}
QLabel#AppTitle {{
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.2px;
    color: {TEXT};
}}
QLabel#AppSubtitle {{ color: {TEXT_FAINT}; font-size: 12px; }}

QFrame#ControlBar {{
    background: rgba(255, 255, 255, 0.62);
    border-bottom: 1px solid {BORDER};
}}

/* ---------- navigation ---------- */
QListWidget#Nav {{
    background: {SIDEBAR};
    border: none;
    border-right: 1px solid {BORDER};
    outline: none;
    padding: 10px 0px;
}}
QListWidget#Nav::item {{
    padding: 10px 14px;
    margin: 3px 10px;
    border-radius: 10px;
    color: {TEXT_DIM};
}}
QListWidget#Nav::item:selected {{
    background: {PRIMARY};
    color: #ffffff;
    font-weight: 600;
}}
QListWidget#Nav::item:hover:!selected {{
    background: rgba(255, 255, 255, 0.85);
    color: {TEXT};
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 7px 15px;
    color: {TEXT};
    font-weight: 500;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT_DEEP}; }}
QPushButton:pressed {{ background: {ACCENT_SOFT}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; background: rgba(255,255,255,0.5); }}

QPushButton#Primary {{
    background: {PRIMARY};
    border: 1px solid {PRIMARY};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: {PRIMARY_DEEP}; border-color: {PRIMARY_DEEP}; color: #ffffff; }}
QPushButton#Primary:disabled {{ background: {PRIMARY_SOFT}; border-color: {PRIMARY_SOFT}; color: #ffffff; }}

QPushButton#Danger {{ border-color: {{rgba_border}}; color: {MISS}; }}
QPushButton#Danger:hover {{ background: {MISS}; border-color: {MISS}; color: #ffffff; }}

/* ---------- inputs ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
    padding: 4px;
}}
QCheckBox {{ color: {TEXT_DIM}; spacing: 7px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 5px;
    background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT_DEEP}; }}

QSlider::groove:horizontal {{
    height: 5px; background: {BORDER}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {SURFACE}; border: 2px solid {ACCENT}; border-radius: 8px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}

QProgressBar {{
    background: {PANEL_ALT};
    border: none;
    border-radius: 5px;
    height: 7px;
    text-align: center;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

/* ---------- menus ---------- */
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 6px;
}}
QMenu::item {{
    padding: 7px 26px 7px 24px;
    border-radius: 7px;
    color: {TEXT};
}}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}

/* Styling a QMenu suppresses the native check mark, so the indicator has to
   be drawn here or a checkable item gives no sign of being on. */
QMenu::indicator {{
    width: 14px;
    height: 14px;
    left: 7px;
    border-radius: 4px;
}}
QMenu::indicator:non-exclusive:unchecked {{
    background: {SURFACE};
    border: 1px solid {BORDER_LIGHT};
}}
QMenu::indicator:non-exclusive:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT_DEEP};
}}

/* ---------- tables ---------- */
QTableWidget, QTableView {{
    background: rgba(255, 255, 255, 0.72);
    alternate-background-color: rgba(241, 245, 249, 0.75);
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 10px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 600;
}}
QHeaderView::section:first {{ border-top-left-radius: 10px; }}
QHeaderView::section:last {{ border-top-right-radius: 10px; }}
QTableWidget::item {{ padding: 6px 9px; }}
QTableCornerButton::section {{ background: {PANEL_ALT}; border: none; }}

QTextEdit, QPlainTextEdit {{
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid {BORDER};
    border-radius: 10px;
    font-family: {MONO};
    font-size: 12px;
    color: {TEXT};
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_LIGHT}; border-radius: 5px; min-height: 34px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_LIGHT}; border-radius: 5px; min-width: 34px; }}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}

/* ---------- misc ---------- */
QStatusBar {{
    background: {GLASS_HEADER};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: {TEXT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 9px;
}}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 12px; }}
QSplitter::handle:vertical {{ height: 12px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 10px; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 7px 15px;
    border: none;
    border-radius: 8px;
    margin-right: 3px;
}}
QTabBar::tab:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_DEEP}; font-weight: 600; }}
"""


def rgba(color: str, alpha: float) -> str:
    """Translucent form of a hex colour.

    Qt stylesheets read an eight-digit hex literal as #AARRGGBB, not #RRGGBBAA,
    so appending an alpha suffix to a colour silently produces a different hue.
    Always build translucent colours through this helper.
    """
    value = color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.3f})"


def badge_style(color: str) -> str:
    """Inline style for a small status pill."""
    return (
        f"background: {rgba(color, 0.12)}; color: {color};"
        f"border: 1px solid {rgba(color, 0.30)};"
        f"border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 600;"
    )


def chip_style(color: str, strong: bool = False) -> str:
    """Inline style for a compact monospaced chip."""
    return (
        f"background: {rgba(color, 0.14 if strong else 0.09)}; color: {color};"
        f"border: 1px solid {rgba(color, 0.34 if strong else 0.22)};"
        f"border-radius: 7px; padding: 4px 9px; font-family: {MONO};"
        f"font-size: 11px; font-weight: {'700' if strong else '500'};"
    )


def series_color(index: int) -> str:
    """Stable categorical colour for the *index*-th series on a chart."""
    return SERIES[index % len(SERIES)]


# ``QPushButton#Danger`` needs a translucent border built through rgba(), which
# cannot be spelled inside the f-string that defines the sheet itself.
STYLESHEET = STYLESHEET.replace("{rgba_border}", rgba(MISS, 0.45))


__all__ = [
    "STYLESHEET", "badge_style", "chip_style", "rgba", "series_color",
    "BG", "BG_TOP", "BG_BOTTOM", "SURFACE", "PANEL", "PANEL_ALT", "SIDEBAR",
    "GLASS", "GLASS_STRONG", "GLASS_HEADER", "BORDER", "BORDER_LIGHT",
    "TEXT", "TEXT_DIM", "TEXT_FAINT",
    "ACCENT", "ACCENT_DEEP", "ACCENT_SOFT", "PRIMARY", "PRIMARY_DEEP", "PRIMARY_SOFT",
    "HIT", "MISS", "FALSE_ALARM", "QUIET", "ACTIVE_TRUTH", "WARN", "DANGER",
    "OUTCOME_COLORS", "ADAPTIVE_COLORS", "SERIES", "AXIS", "GRID_ALPHA",
    "MONO", "SANS",
]
