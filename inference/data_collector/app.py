import sys

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication

from data_collector.core.app_controller import AppController
from data_collector.ui.system_tray import SystemTray


class SystemTrayApp(QObject):
    """
    The main application class that orchestrates the system tray UI and the backend workers.
    """

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        # Prevent the app from closing when a window is closed (important for tray apps)
        self.app.setQuitOnLastWindowClosed(False)

        self.tray = SystemTray()
        self.controller = AppController()

        self.tray.start_collection_requested.connect(self.on_start_request)
        self.tray.stop_collection_requested.connect(self.on_stop_request)

        self.app.aboutToQuit.connect(self.controller.cleanup)

        print("Application ready. Click the tray icon to start.")

    def run(self):
        """Starts the Qt event loop."""
        return self.app.exec()

    def on_start_request(self):
        self.controller.start_collection()
        self.tray.set_menu_state(is_collecting=True)

    def on_stop_request(self):
        self.controller.stop_collection()
        self.tray.set_menu_state(is_collecting=False)
