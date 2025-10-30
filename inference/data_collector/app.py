import sys

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication

from data_collector.core.app_controller import AppController, AppState
from data_collector.ui.main_window import MainWindow


class GazeDataCollectionApp(QObject):
    """
    Main application class with guided UI for gaze data collection.
    """

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)

        self.window = MainWindow()
        self.controller = AppController()

        # Timer for collection elapsed time
        self.collection_timer = QTimer()
        self.collection_timer.timeout.connect(self._update_collection_timer)
        self.collection_elapsed = 0

        # Connect UI signals to controller
        self.window.benchmark_requested.connect(self.controller.start_session)
        self.window.calibration_requested.connect(self.controller.start_data_collection)
        # Note: collection starts automatically after calibration
        self.window.stop_collection_requested.connect(self._on_stop_collection)
        self.window.export_requested.connect(self._on_export_requested)
        self.window.restart_requested.connect(self._on_restart)

        # Connect controller signals to UI
        self.controller.state_changed.connect(self._on_state_changed)

        self.app.aboutToQuit.connect(self.controller.cleanup)

        print("Application ready.")

    def run(self):
        """Starts the Qt event loop."""
        self.window.show()
        return self.app.exec()

    def _on_state_changed(self, state: AppState):
        """Handle state changes from the controller."""
        if state == AppState.BENCHMARKING:
            self.window.on_benchmark_started()
        elif state == AppState.READY_TO_CALIBRATE:
            self.window.on_benchmark_completed()
        elif state == AppState.CALIBRATING:
            self.window.on_calibration_started()
        elif state == AppState.COLLECTING:
            # Calibration completed, now collecting
            self.window.on_calibration_completed()
            self.window.on_collection_started()
            self.collection_elapsed = 0
            self.collection_timer.start(1000)  # Update every second

    def _on_stop_collection(self):
        """Handle stop collection request."""
        self.collection_timer.stop()
        self.controller.stop_session()
        self.window.on_collection_stopped()

    def _update_collection_timer(self):
        """Update the collection timer display."""
        self.collection_elapsed += 1
        self.window.on_collection_time_update(self.collection_elapsed)

    def _on_export_requested(self):
        """Handle upload request (zips and uploads to R2)."""
        import os

        from PyQt6.QtWidgets import QMessageBox

        from data_collector import config
        from data_collector.utils.r2_uploader import R2UploadManager

        self.window.on_export_started()

        # Zip the session data
        zip_path = self.controller.data_manager.export_session_as_zip()

        if not zip_path or not zip_path.exists():
            QMessageBox.critical(self.window, "Error", "Failed to create ZIP file.")
            return

        # Get R2 credentials from environment
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

        if not access_key or not secret_key:
            QMessageBox.critical(
                self.window,
                "Upload Error",
                "R2 credentials not found.\n\n"
                "Please set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables.",
            )
            return

        # Upload to R2
        try:
            uploader = R2UploadManager(
                access_key,
                secret_key,
                config.R2_ENDPOINT_URL,
                config.R2_BUCKET_NAME,
            )

            # Authenticate
            if not uploader.authenticate():
                QMessageBox.critical(
                    self.window,
                    "Upload Error",
                    "Failed to authenticate with R2.\n\n"
                    "Please check your credentials and bucket configuration.",
                )
                return

            # Upload file
            print(f"Uploading {zip_path.name} to R2...")
            file_url = uploader.upload_file(str(zip_path))

            if file_url:
                QMessageBox.information(
                    self.window,
                    "Upload Successful",
                    f"Data uploaded successfully!\n\n"
                    f"File: {zip_path.name}\n"
                    f"Uploaded to R2 bucket: {config.R2_BUCKET_NAME}",
                )
                self.window.on_export_completed()
            else:
                QMessageBox.critical(
                    self.window, "Upload Error", "Failed to upload file to R2."
                )

        except Exception as e:
            QMessageBox.critical(
                self.window, "Upload Error", f"Upload failed:\n{str(e)}"
            )

    def _on_restart(self):
        """Handle restart request."""
        self.collection_timer.stop()
        self.collection_elapsed = 0
        self.controller.cleanup()
        # Recreate controller
        self.controller = AppController()
        self.controller.state_changed.connect(self._on_state_changed)
