import sys

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from data_collector.core.app_controller import AppController
from data_collector.ui.menu_bar_app import MenuBarApp


class UploadWorker(QThread):
    """Worker thread for uploading files to R2 with progress reporting."""

    progress_updated = pyqtSignal(int, int)  # current_bytes, total_bytes
    upload_completed = pyqtSignal(str)  # file_url
    upload_failed = pyqtSignal(str)  # error_message

    def __init__(self, uploader, file_path):
        super().__init__()
        self.uploader = uploader
        self.file_path = file_path

    def run(self):
        """Execute upload in background thread."""
        try:

            def progress_callback(current, total):
                self.progress_updated.emit(current, total)

            file_url = self.uploader.upload_file(self.file_path, progress_callback)

            if file_url:
                self.upload_completed.emit(file_url)
            else:
                self.upload_failed.emit("Upload failed - no URL returned")

        except Exception as e:
            self.upload_failed.emit(str(e))


class GazeDataCollectionApp(QObject):
    """
    Main application class with menu bar interface for gaze data collection.
    """

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.controller = AppController()
        self.menu_bar = MenuBarApp(controller=self.controller)
        self.menu_bar.restart_requested.connect(self._on_restart)
        self.app.aboutToQuit.connect(self._on_quit)

        print("Application ready.")

    def run(self):
        """Starts the Qt event loop."""
        return self.app.exec()

    def _on_restart(self):
        """Handle restart request from menu bar."""
        self.controller.cleanup()

        # Recreate controller
        self.controller = AppController()

        # Update menu bar reference
        self.menu_bar.controller = self.controller
        self.controller.state_changed.connect(self.menu_bar._on_state_changed)

    def _on_quit(self):
        """Cleanup on quit."""
        self.controller.cleanup()
        self.menu_bar.cleanup()
