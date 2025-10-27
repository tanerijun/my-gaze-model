import sys

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication

from data_collector.core.app_controller import AppController, AppState
from data_collector.ui.system_tray import SystemTray


class SystemTrayApp(QObject):
    """
    The main application class that orchestrates the system tray UI and the backend workers.
    """

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.tray = SystemTray()
        self.controller = AppController()

        self.tray.start_requested.connect(self.on_start_request)
        self.tray.stop_requested.connect(self.controller.stop_session)

        self.controller.state_changed.connect(self.tray.update_state)

        self.app.aboutToQuit.connect(self.controller.cleanup)

        print("Application ready. Click the tray icon to start.")

    def run(self):
        """Starts the Qt event loop."""
        return self.app.exec()

    def on_start_request(self):
        """Handles the primary 'start' action from the tray menu."""
        current_state = self.controller.state
        if current_state == AppState.IDLE:
            self.controller.start_session()
        elif current_state == AppState.READY_TO_CALIBRATE:
            self.controller.start_data_collection()
