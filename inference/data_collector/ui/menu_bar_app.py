"""
Menu Bar Application for Gaze Data Collection

Provides a menu bar interface that controls the data collection workflow
"""

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QInputDialog,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSystemTrayIcon,
)

from data_collector import config
from data_collector.core.app_controller import AppController, AppState


class MenuBarApp(QObject):
    """Menu bar application that controls the data collection workflow."""

    restart_requested = pyqtSignal()

    def __init__(self, controller=None):
        super().__init__()

        self.controller = controller if controller else AppController()

        # Timer for collection elapsed time
        self.collection_timer = QTimer()
        self.collection_timer.timeout.connect(self._update_collection_timer)
        self.collection_elapsed = 0

        self._setup_menu_bar()
        self.controller.state_changed.connect(self._on_state_changed)

        print("Menu bar app initialized.")

    def _show_focused_message(
        self, title, message, icon_type=QMessageBox.Icon.Information
    ):
        """Show a message box with forced focus and always-on-top behavior."""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(icon_type)
        msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        msg.activateWindow()
        msg.raise_()
        msg.exec()

    def _show_focused_question(self, title, message, buttons):
        """Show a question dialog with forced focus and always-on-top behavior."""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setStandardButtons(buttons)
        msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        msg.activateWindow()
        msg.raise_()
        return msg.exec()

    def _show_focused_critical(
        self, title, message, buttons=QMessageBox.StandardButton.Ok
    ):
        """Show a critical error dialog with forced focus and always-on-top behavior."""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setStandardButtons(buttons)
        msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        msg.activateWindow()
        msg.raise_()
        return msg.exec()

    def _setup_menu_bar(self):
        """Setup the menu bar icon and menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("System tray is not available on this system.")

        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(QIcon(str(config.ICON_PATH)))
        self.tray_icon.setToolTip("Gaze Data Collection")

        self.menu = QMenu()
        self._create_menu_actions()
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.show()

    def _create_menu_actions(self):
        """Create menu actions for the menu bar."""
        # Start Session
        self.start_action = QAction("🟢 Start Session")
        self.start_action.triggered.connect(self._on_start_session)
        self.menu.addAction(self.start_action)

        # Stop Collection
        self.stop_action = QAction("🔴 Stop Collection")
        self.stop_action.triggered.connect(self._on_stop_collection)
        self.stop_action.setVisible(False)
        self.menu.addAction(self.stop_action)

        self.menu.addSeparator()

        # Restart
        self.restart_action = QAction("🔄 Restart")
        self.restart_action.triggered.connect(self._on_restart)
        self.menu.addAction(self.restart_action)

        # Help
        self.help_action = QAction("❓ Help")
        self.help_action.triggered.connect(self._on_help)
        self.menu.addAction(self.help_action)

        self.menu.addSeparator()

        # Quit
        self.quit_action = QAction("❌ Quit")
        self.quit_action.triggered.connect(self._on_quit)
        self.menu.addAction(self.quit_action)

    def _on_start_session(self):
        """Handle start session request."""
        # Ask for participant name
        name, ok = QInputDialog.getText(
            None,
            "Participant Name",
            "Please enter your name or ID:",
            text="Participant",
        )

        if ok and name.strip():
            self.controller.participant_name = name.strip()
        else:
            self.controller.participant_name = "Anonymous"

        print(f"Participant: {self.controller.participant_name}")

        # Show info message
        self._show_focused_message(
            "Starting Session",
            "System benchmark will start now.\nThis may take a few moments.",
        )

        # Start benchmarking
        self.controller.start_session()

    def _on_state_changed(self, state: AppState):
        """Handle state changes from the controller."""
        if state == AppState.BENCHMARKING:
            self.start_action.setText("⏳ Benchmarking...")
            self.start_action.setEnabled(False)
            self.stop_action.setVisible(False)

        elif state == AppState.READY_TO_CALIBRATE:
            self.start_action.setText("▶️ Start Calibration")
            self.start_action.setEnabled(True)
            self.start_action.disconnect()
            self.start_action.triggered.connect(self._on_start_calibration)
            self.stop_action.setVisible(False)

            self._show_focused_message(
                "Benchmark Complete",
                "System benchmark completed successfully!\n\nClick 'Start Calibration' from the menu to continue.",
            )

        elif state == AppState.CALIBRATING:
            self.start_action.setText("📍 Calibrating...")
            self.start_action.setEnabled(False)
            self.stop_action.setVisible(False)

        elif state == AppState.COLLECTING:
            self.start_action.setText("🔴 Collecting Data - 0m 0s")
            self.start_action.setEnabled(False)
            self.stop_action.setVisible(True)

            self.collection_elapsed = 0
            self.collection_timer.start(1000)

            self._show_focused_message(
                "Collection Started",
                "Calibration completed!\n\nData collection is now active.\n每 20 秒會出現藍點，和上個步驟一樣注視並點擊藍點。\n\nClick 'Stop Collection' from the menu when done.",
            )

    def _on_start_calibration(self):
        """Handle start calibration request."""
        self._show_focused_message(
            "Starting Calibration",
            "Calibration will now begin.\n\n- 頭部保持不動.\n- 注視好藍點後再點擊（別眨眼）\n- 休息（可以眨眼）\n- 按「空白鍵」繼續",
        )
        self.controller.start_data_collection()

    def _on_stop_collection(self):
        """Handle stop collection request."""
        self.collection_timer.stop()
        self.controller.stop_session()

        minutes = self.collection_elapsed // 60
        seconds = self.collection_elapsed % 60

        self.start_action.setText("📦 Export Data")
        self.start_action.setEnabled(True)
        self.start_action.disconnect()
        self.start_action.triggered.connect(self._on_export_requested)
        self.stop_action.setVisible(False)

        self._show_focused_message(
            "Collection Complete",
            f"Data collection stopped.\n\nTotal time: {minutes}m {seconds}s\n\nNext step: Click 'Export Data' from the menu to upload your data and complete the experiment.",
        )

    def _update_collection_timer(self):
        """Update the collection timer."""
        self.collection_elapsed += 1
        minutes = self.collection_elapsed // 60
        seconds = self.collection_elapsed % 60
        # Update both menu text and tooltip
        self.start_action.setText(f"🔴 Collecting Data - {minutes}m {seconds}s")
        self.tray_icon.setToolTip(f"Collecting Data - {minutes}m {seconds}s")

    def _on_export_requested(self):
        """Handle export request."""
        import os

        # Zip the session data
        zip_path = self.controller.data_manager.export_session_as_zip()

        if not zip_path or not zip_path.exists():
            self._show_focused_critical("Error", "Failed to create ZIP file.")
            return

        # Get R2 credentials from environment
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

        if not access_key or not secret_key:
            self._show_focused_critical(
                "Upload Error",
                "R2 credentials not found.\n\n"
                "Please set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables.",
            )
            return

        # Upload to R2
        self._attempt_upload(zip_path, access_key, secret_key)

    def _attempt_upload(self, zip_path, access_key, secret_key, retry_count=0):
        """Attempt to upload with error handling and retry capability."""
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
                reply = self._show_focused_critical(
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
                None,
            )
            progress.setWindowTitle("Uploading to R2")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setWindowFlags(
                Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint
            )
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.activateWindow()
            progress.raise_()
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

            # Create upload worker (reuse from app.py)
            from data_collector.app import UploadWorker

            self._upload_worker = UploadWorker(uploader, str(zip_path))

            # Connect signals
            def update_progress(current, total):
                if upload_cancelled:
                    return
                percentage = int((current / total) * 100) if total > 0 else 0
                progress.setValue(percentage)
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

                # Reset to start for new session
                self.start_action.setText("🟢 Start Session")
                self.start_action.setEnabled(True)
                self.start_action.disconnect()
                self.start_action.triggered.connect(self._on_start_session)

                self._show_focused_message(
                    "Experiment Complete",
                    f"Data uploaded successfully!\n\n"
                    f"File: {zip_path.name}\n"
                    f"Uploaded to server bucket: {config.R2_BUCKET_NAME}\n\n"
                    f"The experiment is now complete. Thank you!",
                )

            def on_failed(error_message):
                if upload_cancelled:
                    progress.close()
                    return
                progress.close()
                reply = self._show_focused_critical(
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

            print(f"Starting upload of {zip_path.name} to server...")
            self._upload_worker.start()

        except Exception as e:
            reply = self._show_focused_critical(
                "Upload Error",
                f"Failed to initialize upload:\n{str(e)}\n\nWould you like to retry?",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Retry:
                self._attempt_upload(zip_path, access_key, secret_key, retry_count + 1)

    def _on_restart(self):
        """Handle restart request."""
        reply = self._show_focused_question(
            "Restart",
            "Are you sure you want to restart?\n\nThis will reset the current session.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.collection_timer.stop()
            self.collection_elapsed = 0

            # Reset menu
            self.start_action.setText("🟢 Start Session")
            self.start_action.setEnabled(True)
            self.start_action.disconnect()
            self.start_action.triggered.connect(self._on_start_session)
            self.stop_action.setVisible(False)
            self.tray_icon.setToolTip("Gaze Data Collection")

            # Emit signal for main app to handle controller restart
            self.restart_requested.emit()

    def _on_help(self):
        """Show help dialog."""
        self._show_focused_message(
            "Help",
            "Gaze Data Collection Tool\n\n"
            "How to use:\n"
            "1. Click 'Start Session' to begin\n"
            "2. Wait for benchmark to complete\n"
            "3. Start calibration, fixate, and click on points (keep head still)\n"
            "4. Data collection will start automatically\n"
            "5. Stop collection when done\n"
            "6. Upload data to server\n\n"
            "The calibration overlay will appear every 20 seconds during collection.\n"
            "Just click on the points just like during the calibration process.\n\n"
            "---\n\n"
            "使用方法：\n"
            "1. 點擊「Start Session」\n"
            "2. 等待基本資料搜集完成\n"
            "3. 開始矯正，注視並點擊藍點（頭部保持不動）\n"
            "4. 資料收集將自動開始\n"
            "5. 完成後點 「Stop collection」\n"
            "6. 將資料上傳到伺服器\n\n"
            "在收集期間，每 20 秒出現一次藍點，出現時注視著藍點並點擊。\n"
            "---\n\n"
            "FAQ:\n\n"
            "Q: Application freeze/crash/hang!\n"
            "A: This usually happens when granting new permission access. Just RESTART the app!\n\n"
            "Q: What if I missed the permission granting form?\n"
            "A: Give permission manually.\n"
            "\tIn MacOS: Settings -> Privacy & Security -> add app to Accessibility, Input Monitoring, Screen & System Audio Recording",
        )

    def _on_quit(self):
        """Quit the application."""
        reply = self._show_focused_question(
            "Quit",
            "Are you sure you want to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.collection_timer.stop()
            self.controller.cleanup()
            QApplication.quit()

    def cleanup(self):
        """Cleanup resources."""
        self.collection_timer.stop()
        self.tray_icon.hide()
