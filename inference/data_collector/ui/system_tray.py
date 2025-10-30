from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from data_collector import config
from data_collector.core.app_controller import AppState


class SystemTray(QObject):
    """Manages the system tray icon and its menu."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("System tray is not available on this system.")

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(str(config.ICON_PATH)))
        self.tray_icon.setToolTip("Gaze Tracker")

        self.menu = QMenu()
        self._create_actions()
        self.tray_icon.setContextMenu(self.menu)

        self.tray_icon.show()
        print("System tray icon initialized.")

    def _create_actions(self):
        """Creates the menu actions."""
        self.main_action = QAction("🟢 Start Session")
        self.main_action.triggered.connect(self.start_requested.emit)
        self.menu.addAction(self.main_action)

        self.stop_action = QAction("🔴 Stop Session")
        self.stop_action.triggered.connect(self.stop_requested.emit)
        self.stop_action.setVisible(False)
        self.menu.addAction(self.stop_action)

        self.menu.addSeparator()
        self.about_action = QAction("ℹ️ About")
        self.menu.addAction(self.about_action)
        self.quit_action = QAction("❌ Quit")
        self.quit_action.triggered.connect(QApplication.quit)
        self.menu.addAction(self.quit_action)

    @pyqtSlot(AppState)
    def update_state(self, state: AppState):
        """Updates the menu based on the application's state."""
        is_active = state not in [AppState.IDLE, AppState.READY_TO_CALIBRATE]
        self.stop_action.setVisible(is_active)

        if state == AppState.IDLE:
            self.main_action.setText("🟢 Start Session")
            self.main_action.setEnabled(True)
        elif state == AppState.BENCHMARKING:
            self.main_action.setText("⏳ Benchmarking...")
            self.main_action.setEnabled(False)
        elif state == AppState.READY_TO_CALIBRATE:
            self.main_action.setText("▶️ Start Calibration")
            self.main_action.setEnabled(True)
        elif state == AppState.CALIBRATING:
            self.main_action.setText("📍 Calibrating...")
            self.main_action.setEnabled(False)
        elif state == AppState.COLLECTING:
            self.main_action.setText("🔴 Collecting Data...")
            self.main_action.setEnabled(False)
        elif state == AppState.PAUSED_BY_DRIFT:
            self.main_action.setText("⚠️ Paused (Head Pose Drift)")
            self.main_action.setEnabled(False)
