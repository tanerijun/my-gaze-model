import sys

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication

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

        self.tray.start_collection_requested.connect(self.start_collection)
        self.tray.stop_collection_requested.connect(self.stop_collection)

        print("Application ready. Click the tray icon to start.")

    def run(self):
        """Starts the Qt event loop."""
        return self.app.exec()

    def start_collection(self):
        """
        Placeholder slot for starting the data collection process.
        In the next step, this will initialize and start the worker threads.
        """
        print("[APP LOGIC] 'Start Collection' signal received. Starting process...")
        self.tray.set_menu_state(is_collecting=True)

    def stop_collection(self):
        """
        Placeholder slot for stopping the data collection process.
        """
        print("[APP LOGIC] 'Stop Collection' signal received. Stopping process...")
        self.tray.set_menu_state(is_collecting=False)
