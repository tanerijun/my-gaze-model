import sys

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from data_collector.core.app_controller import AppController, AppState
from data_collector.ui.main_window import MainWindow


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
        self.window.benchmark_requested.connect(self._on_benchmark_requested)
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

    def _on_benchmark_requested(self):
        """Handle benchmark request - store participant name and start session."""
        # Store participant name from window
        self.controller.participant_name = self.window.participant_name
        self.controller.start_session()

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
        self._attempt_upload(zip_path, access_key, secret_key)

    def _attempt_upload(self, zip_path, access_key, secret_key, retry_count=0):
        """Attempt to upload with error handling and retry capability."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog

        from data_collector import config
        from data_collector.utils.r2_uploader import R2UploadManager

        try:
            uploader = R2UploadManager(
                access_key,
                secret_key,
                config.R2_ENDPOINT_URL,
                config.R2_BUCKET_NAME,
            )

            # Authenticate
            if not uploader.authenticate():
                reply = QMessageBox.critical(
                    self.window,
                    "Upload Error",
                    "Failed to authenticate with R2.\n\n"
                    "Please check your credentials and bucket configuration.\n\n"
                    "Would you like to retry?",
                    QMessageBox.StandardButton.Retry
                    | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Retry:
                    self._attempt_upload(
                        zip_path, access_key, secret_key, retry_count + 1
                    )
                return

            # Show progress dialog
            progress = QProgressDialog(
                "Preparing upload...",
                "Cancel",
                0,
                100,
                self.window,
            )
            progress.setWindowTitle("Uploading to R2")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            # Track if upload was cancelled
            upload_cancelled = False

            def on_cancel():
                nonlocal upload_cancelled
                upload_cancelled = True
                if hasattr(self, "_upload_worker") and self._upload_worker.isRunning():
                    self._upload_worker.terminate()
                    self._upload_worker.wait()

            progress.canceled.connect(on_cancel)

            # Create and start upload worker
            self._upload_worker = UploadWorker(uploader, str(zip_path))

            # Connect signals
            def update_progress(current, total):
                if upload_cancelled:
                    return
                percentage = int((current / total) * 100) if total > 0 else 0
                progress.setValue(percentage)
                # Format bytes nicely
                current_mb = current / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                progress.setLabelText(
                    f"Uploading {zip_path.name}...\n"
                    f"{current_mb:.1f} MB / {total_mb:.1f} MB ({percentage}%)"
                )

            def on_completed(file_url):
                if upload_cancelled:
                    return
                progress.close()
                QMessageBox.information(
                    self.window,
                    "Upload Successful",
                    f"Data uploaded successfully!\n\n"
                    f"File: {zip_path.name}\n"
                    f"Uploaded to R2 bucket: {config.R2_BUCKET_NAME}",
                )
                self.window.on_export_completed()

            def on_failed(error_message):
                if upload_cancelled:
                    progress.close()
                    return
                progress.close()
                reply = QMessageBox.critical(
                    self.window,
                    "Upload Error",
                    f"Upload failed:\n{error_message}\n\nWould you like to retry?",
                    QMessageBox.StandardButton.Retry
                    | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Retry:
                    self._attempt_upload(
                        zip_path, access_key, secret_key, retry_count + 1
                    )

            self._upload_worker.progress_updated.connect(update_progress)
            self._upload_worker.upload_completed.connect(on_completed)
            self._upload_worker.upload_failed.connect(on_failed)

            print(f"Starting upload of {zip_path.name} to R2...")
            self._upload_worker.start()

        except Exception as e:
            reply = QMessageBox.critical(
                self.window,
                "Upload Error",
                f"Failed to initialize upload:\n{str(e)}\n\nWould you like to retry?",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Retry:
                self._attempt_upload(zip_path, access_key, secret_key, retry_count + 1)

    def _on_restart(self):
        """Handle restart request."""
        self.collection_timer.stop()
        self.collection_elapsed = 0
        self.controller.cleanup()
        # Recreate controller
        self.controller = AppController()
        self.controller.state_changed.connect(self._on_state_changed)
