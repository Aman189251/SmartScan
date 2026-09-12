"""Main application window.

A native technical workstation: an identity bar, a run control strip, four
working views, and a status line that always says what the engine and the model
are doing.  Polling is split by cost so the interface stays responsive: cheap
status at a few hertz, heavier snapshots less often, and the activity grid least
often.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .api_client import ApiClient, ApiError
from .views.dashboard_view import DashboardView
from .views.experiment_view import ExperimentView
from .views.learning_view import LearningView
from .views.models_view import ModelsView
from .widgets.cards import Badge, ToggleSwitch

NAV_ITEMS = [
    ("Dashboard", "grid", "Live scan decisions and outcomes"),
    ("Learning monitor", "pulse", "Offline model and the hidden adaptive layer"),
    ("Experiment arena", "bars", "Compare policies on identical environments"),
    ("Models and runs", "layers", "Registry, training and run history"),
]


def _nav_pixmap(kind: str, color: str, size: int = 18) -> QPixmap:
    """Draw a navigation glyph, so no icon font has to be installed."""
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    qcolor = QColor(color)
    pen = QPen(qcolor)
    pen.setWidthF(1.8 * scale)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    s = size * scale

    if kind == "grid":
        painter.setBrush(qcolor)
        for x, y, w, h in ((0.16, 0.16, 0.32, 0.42), (0.54, 0.16, 0.30, 0.24),
                           (0.16, 0.64, 0.32, 0.20), (0.54, 0.46, 0.30, 0.38)):
            painter.drawRoundedRect(s * x, s * y, s * w, s * h, 2.4 * scale, 2.4 * scale)
    elif kind == "pulse":
        path = QPainterPath()
        path.moveTo(s * 0.12, s * 0.56)
        path.lineTo(s * 0.32, s * 0.56)
        path.lineTo(s * 0.44, s * 0.24)
        path.lineTo(s * 0.58, s * 0.78)
        path.lineTo(s * 0.68, s * 0.50)
        path.lineTo(s * 0.88, s * 0.50)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
    elif kind == "bars":
        painter.setBrush(qcolor)
        for x, top in ((0.18, 0.52), (0.42, 0.26), (0.66, 0.40)):
            painter.drawRoundedRect(s * x, s * top, s * 0.16, s * (0.84 - top),
                                    2.0 * scale, 2.0 * scale)
    elif kind == "layers":
        painter.setBrush(Qt.NoBrush)
        for y in (0.22, 0.46, 0.70):
            path = QPainterPath()
            path.moveTo(s * 0.50, s * (y - 0.10))
            path.lineTo(s * 0.86, s * y)
            path.lineTo(s * 0.50, s * (y + 0.10))
            path.lineTo(s * 0.14, s * y)
            path.closeSubpath()
            painter.drawPath(path)

    painter.end()
    return pixmap


def _compact_combo(min_chars: int) -> QComboBox:
    """Combo box that shrinks instead of demanding room for its longest item.

    A default QComboBox reports a minimum width wide enough for every entry it
    holds.  Several of those in one strip can push the row's minimum past the
    window, and Qt resolves that by overlapping widgets rather than clipping
    them, which puts labels on top of the controls beside them.
    """
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    return combo


def _nav_icon(kind: str) -> QIcon:
    """Icon that turns white once its row is the selected one."""
    icon = QIcon()
    icon.addPixmap(_nav_pixmap(kind, theme.TEXT_DIM), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_nav_pixmap(kind, "#ffffff"), QIcon.Selected, QIcon.On)
    icon.addPixmap(_nav_pixmap(kind, "#ffffff"), QIcon.Selected, QIcon.Off)
    return icon


class MainWindow(QMainWindow):
    def __init__(self, client: ApiClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Smart Scan Strategy - Adaptive ES Receiver Scheduler")
        self.resize(1560, 950)
        self.setMinimumSize(QSize(1180, 760))

        central = QWidget()
        central.setObjectName("Page")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.setFixedWidth(218)
        self.nav.setIconSize(QSize(18, 18))
        for title, icon_kind, tip in NAV_ITEMS:
            item = QListWidgetItem(_nav_icon(icon_kind), title)
            item.setToolTip(tip)
            self.nav.addItem(item)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._on_nav)
        body.addWidget(self.nav)

        # A container widget rather than a bare nested layout: the control bar
        # is a styled QFrame, and giving it a real parent widget keeps its
        # geometry from being resolved against the outer layout, which left it
        # drawn on top of the view underneath.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._build_control_bar())

        self.stack = QStackedWidget()
        self.dashboard = DashboardView()
        self.learning = LearningView()
        self.experiments = ExperimentView()
        self.models = ModelsView()
        for view in (self.dashboard, self.learning, self.experiments, self.models):
            self.stack.addWidget(view)
        right_layout.addWidget(self.stack, 1)
        body.addWidget(right, 1)
        root.addLayout(body, 1)

        self.setStatusBar(QStatusBar())
        self.status_message = QLabel("Connecting to the local service...")
        self.statusBar().addWidget(self.status_message, 1)
        self.scope_label = QLabel("synthetic research simulator  -  offline")
        self.scope_label.setObjectName("Faint")
        self.statusBar().addPermanentWidget(self.scope_label)

        self.experiments.run_requested.connect(self._start_experiment)
        self.experiments.load_requested.connect(self._load_experiment)
        self.models.train_requested.connect(self._start_training)
        self.models.promote_requested.connect(self._promote_model)
        self.models.refresh_requested.connect(self._refresh_importance)

        self._experiment_job: Optional[str] = None
        self._training_job: Optional[str] = None
        self._last_run_id: Optional[str] = None

        self._load_catalogues()
        self._refresh_models()

        self.timer_status = QTimer(self)
        self.timer_status.timeout.connect(self._poll_status)
        self.timer_status.start(250)

        self.timer_snapshot = QTimer(self)
        self.timer_snapshot.timeout.connect(self._poll_snapshot)
        self.timer_snapshot.start(600)

        self.timer_slow = QTimer(self)
        self.timer_slow.timeout.connect(self._poll_slow)
        self.timer_slow.start(2000)

    # ------------------------------------------------------------------ header
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(66)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(14)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Smart Scan Strategy")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "Adaptive machine-learning scan scheduling  -  synthetic research simulator")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addStretch(1)

        self.model_badge = Badge("model: --", theme.TEXT_DIM)
        self.model_badge.setToolTip("The offline model backing the current run.")
        layout.addWidget(self.model_badge)
        return header

    def _build_control_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ControlBar")
        bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 9, 16, 9)
        layout.setSpacing(9)

        layout.addWidget(self._label("Scenario"))
        self.scenario_combo = _compact_combo(14)
        layout.addWidget(self.scenario_combo, 1)

        layout.addWidget(self._label("Strategy"))
        self.strategy_combo = _compact_combo(12)
        layout.addWidget(self.strategy_combo, 1)

        layout.addWidget(self._label("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999_999)
        self.seed_spin.setValue(20260909 % 999_999)
        self.seed_spin.setMaximumWidth(96)
        layout.addWidget(self.seed_spin)

        layout.addSpacing(4)
        self.adaptive_toggle = ToggleSwitch(checked=True)
        self.adaptive_toggle.setToolTip(
            "Runs the hidden adaptive learner. It starts in shadow mode and only gains "
            "influence once its acceptance criteria are met.")
        layout.addWidget(self.adaptive_toggle)
        adaptive_label = self._label("Adaptive layer")
        adaptive_label.setToolTip(self.adaptive_toggle.toolTip())
        layout.addWidget(adaptive_label)

        speed_box = QVBoxLayout()
        speed_box.setSpacing(1)
        self.speed_label = QLabel("Speed  40 slots/s")
        self.speed_label.setObjectName("Faint")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200)
        self.speed_slider.setValue(40)
        self.speed_slider.setFixedWidth(112)
        self.speed_slider.valueChanged.connect(self._on_speed)
        speed_box.addWidget(self.speed_label)
        speed_box.addWidget(self.speed_slider)
        layout.addLayout(speed_box)

        layout.addStretch(1)

        self.start_button = QPushButton("Start run")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._start_run)
        layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self._stop_run)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        return bar

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Dim")
        return label

    # --------------------------------------------------------------- catalogue
    def _load_catalogues(self) -> None:
        try:
            scenarios = self.client.scenarios()
            strategies = self.client.strategies()
        except ApiError as exc:
            self._set_status(f"Service not reachable: {exc}", theme.DANGER)
            return
        self.scenario_combo.clear()
        for scenario in scenarios:
            self.scenario_combo.addItem(scenario.get("title", scenario["key"]), scenario["key"])
            self.scenario_combo.setItemData(
                self.scenario_combo.count() - 1, scenario.get("description", ""), Qt.ToolTipRole)
        index = self.scenario_combo.findData("mixed")
        if index >= 0:
            self.scenario_combo.setCurrentIndex(index)

        self.strategy_combo.clear()
        for strategy in strategies:
            label = strategy.get("title", strategy["key"])
            self.strategy_combo.addItem(label, strategy["key"])
        index = self.strategy_combo.findData("smart_scan")
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)

        self.experiments.set_scenarios(scenarios)

    # ------------------------------------------------------------------ control
    def _start_run(self) -> None:
        try:
            self.client.start(
                strategy=self.strategy_combo.currentData() or "smart_scan",
                scenario=self.scenario_combo.currentData() or "mixed",
                seed=int(self.seed_spin.value()),
                adaptive=bool(self.adaptive_toggle.isChecked()),
                speed=float(self.speed_slider.value()),
                persist=True,
            )
            self.learning.reset()
            self._set_status("Run started", theme.HIT)
        except ApiError as exc:
            self._set_status(str(exc), theme.DANGER)

    def _toggle_pause(self) -> None:
        try:
            if self.pause_button.text() == "Pause":
                self.client.pause()
            else:
                self.client.resume()
        except ApiError as exc:
            self._set_status(str(exc), theme.DANGER)

    def _stop_run(self) -> None:
        try:
            self.client.stop()
            self._set_status("Run stopped", theme.TEXT_DIM)
        except ApiError as exc:
            self._set_status(str(exc), theme.DANGER)

    def _on_speed(self, value: int) -> None:
        self.speed_label.setText(
            "Speed  unlimited" if value == 0 else f"Speed  {value} slots/s")
        try:
            self.client.set_speed(float(value))
        except ApiError:
            pass

    def _on_nav(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        if row == 3:
            self._refresh_models()
        elif row == 2:
            self._refresh_experiment_history()

    # ------------------------------------------------------------------ polling
    def _poll_status(self) -> None:
        try:
            status = self.client.status()
        except ApiError as exc:
            self._set_status(f"Service unavailable: {exc}", theme.DANGER)
            return
        self.dashboard.update_status(status)

        state = str(status.get("state", "IDLE"))
        running = state == "RUNNING"
        paused = state == "PAUSED"
        self.start_button.setEnabled(not running and not paused)
        self.pause_button.setEnabled(running or paused)
        self.pause_button.setText("Resume" if paused else "Pause")
        self.stop_button.setEnabled(running or paused)

        run_id = status.get("run_id")
        if run_id and run_id != self._last_run_id:
            self._last_run_id = run_id
            self.learning.reset()

        model = status.get("model", {}) or {}
        kind = model.get("kind", "--")
        trained = bool(model.get("trained"))
        self.model_badge.set_state(
            f"model: {kind}" + (f"  ({model.get('model_id', '')[:24]})" if trained
                                else "  (untrained heuristic)"),
            theme.HIT if trained else theme.WARN)
        if state in ("RUNNING", "PAUSED"):
            self._set_status(
                f"{status.get('strategy', '')} on {status.get('scenario', '')}  -  "
                f"slot {status.get('slot', 0):,} of {status.get('n_slots', 0):,}  -  "
                f"rolling detection rate {float(status.get('rolling_detection_rate', 0)):.3f}",
                theme.TEXT_DIM)

    def _poll_snapshot(self) -> None:
        if self.stack.currentIndex() > 1:
            return
        try:
            snapshot = self.client.snapshot()
        except ApiError:
            return
        if snapshot.get("state") in (None, "IDLE"):
            return
        self.dashboard.update_snapshot(snapshot)
        adaptive = snapshot.get("adaptive", {}) or {}
        if adaptive:
            self.learning.update_adaptive(adaptive)

    def _poll_slow(self) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            try:
                status = self.client.status()
                payload = self.client.activity(width=240)
                current = (status.get("current") or {}).get("band")
                self.dashboard.update_activity(payload, current)
            except ApiError:
                pass
        elif index == 1:
            try:
                self.learning.update_model(self.client.models())
                run_id = self._last_run_id
                if run_id:
                    detail = self.client.run_detail(run_id)
                    self.learning.add_events(detail.get("adaptive_events", []))
            except ApiError:
                pass
        self._poll_jobs()

    def _poll_jobs(self) -> None:
        for job_id, kind in ((self._experiment_job, "experiment"),
                             (self._training_job, "training")):
            if not job_id:
                continue
            try:
                job = self.client.job(job_id)
            except ApiError:
                continue
            busy = job["status"] in ("PENDING", "RUNNING")
            message = job.get("message") or job["status"]
            progress = float(job.get("progress", 0.0))
            if kind == "experiment":
                self.experiments.set_busy(busy, message, progress)
                if not busy:
                    self._experiment_job = None
                    if job["status"] == "FINISHED" and job.get("result"):
                        self.experiments.set_result(job["result"])
                        self._refresh_experiment_history()
                    elif job.get("error"):
                        self._warn("Experiment failed", job["error"])
            else:
                self.models.set_busy(busy, message, progress)
                if not busy:
                    self._training_job = None
                    if job["status"] == "FINISHED":
                        self._refresh_models()
                        self._set_status("Training complete; new model promoted", theme.HIT)
                    elif job.get("error"):
                        self._warn("Training failed", job["error"])

    # ------------------------------------------------------------------ actions
    def _start_experiment(self, params: Dict[str, Any]) -> None:
        try:
            job = self.client.run_experiment(**params)
            self._experiment_job = job["job_id"]
            self.experiments.set_busy(True, "starting", 0.0)
        except ApiError as exc:
            self._warn("Could not start experiment", str(exc))

    def _load_experiment(self, name: str) -> None:
        try:
            self.experiments.set_result(self.client.experiment_file(name))
        except ApiError as exc:
            self._warn("Could not load experiment", str(exc))

    def _refresh_experiment_history(self) -> None:
        try:
            self.experiments.set_history(self.client.experiment_files())
        except ApiError:
            pass

    def _start_training(self, params: Dict[str, Any]) -> None:
        try:
            job = self.client.train(**params)
            self._training_job = job["job_id"]
            self.models.set_busy(True, "starting", 0.0)
        except ApiError as exc:
            self._warn("Could not start training", str(exc))

    def _promote_model(self, model_id: str) -> None:
        try:
            self.client.promote_model(model_id)
            self._refresh_models()
            self._set_status(f"{model_id} promoted; it loads on the next run", theme.HIT)
        except ApiError as exc:
            self._warn("Could not promote model", str(exc))

    def _refresh_models(self) -> None:
        try:
            payload = self.client.models()
            self.models.set_models(payload)
            self.learning.update_model(payload)
            self.models.set_runs(self.client.runs(limit=25))
        except ApiError:
            pass

    def _refresh_importance(self) -> None:
        model_id = self.models.selected_model()
        if not model_id:
            return
        try:
            detail = self.client.model_detail(model_id)
            self.models.set_importance(detail.get("feature_importance", {}) or {})
        except ApiError:
            pass

    # -------------------------------------------------------------------- misc
    def _set_status(self, message: str, color: str = theme.TEXT_DIM) -> None:
        self.status_message.setText(message)
        self.status_message.setStyleSheet(f"color: {color};")

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message[:1200])

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for timer in (self.timer_status, self.timer_snapshot, self.timer_slow):
            timer.stop()
        try:
            self.client.stop()
        except ApiError:
            pass
        super().closeEvent(event)


__all__ = ["MainWindow"]
