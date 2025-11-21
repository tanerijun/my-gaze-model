"""
Main Window UI for Gaze Data Collection

A simple, guided interface for non-technical users to collect gaze data.
Shows a clear step-by-step workflow with visual progress indicators.
"""

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class StepWidget(QWidget):
    """A single step in the workflow with status indicator and action button."""

    clicked = pyqtSignal()

    def __init__(self, step_number: int, title: str, description: str, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.title = title
        self.description = description
        self.is_completed = False
        self.is_active = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Step indicator (number or checkmark)
        self.indicator = QLabel()
        self.indicator.setFixedSize(40, 40)
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator_font = QFont()
        indicator_font.setPointSize(16)
        indicator_font.setBold(True)
        self.indicator.setFont(indicator_font)
        self.indicator.setStyleSheet(
            """
            QLabel {
                background-color: #cccccc;
                color: #666666;
                border-radius: 20px;
                border: 2px solid #999999;
            }
        """
        )
        layout.addWidget(self.indicator)

        # Step content (title and description)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        self.title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.desc_label = QLabel(self.description)
        self.desc_label.setStyleSheet("color: #666666;")
        desc_font = QFont()
        desc_font.setPointSize(11)
        self.desc_label.setFont(desc_font)

        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.desc_label)
        layout.addLayout(content_layout, 1)

        # Action button
        self.action_button = QPushButton("Start")
        self.action_button.setFixedWidth(120)
        self.action_button.setFixedHeight(35)
        button_font = QFont()
        button_font.setPointSize(12)
        self.action_button.setFont(button_font)
        self.action_button.clicked.connect(self.clicked.emit)
        layout.addWidget(self.action_button)

        # Start in pending state
        self.set_pending()

    def set_pending(self):
        """Step is waiting to be started."""
        self.is_completed = False
        self.is_active = False
        self.indicator.setText(str(self.step_number))
        self.indicator.setStyleSheet(
            """
            QLabel {
                background-color: #e0e0e0;
                color: #999999;
                border-radius: 20px;
                border: 2px solid #cccccc;
            }
        """
        )
        self.title_label.setStyleSheet("color: #999999;")
        self.desc_label.setStyleSheet("color: #cccccc;")
        self.action_button.setEnabled(False)
        self.action_button.setStyleSheet("")

    def set_ready(self):
        """Step is ready to be started by the user."""
        self.is_completed = False
        self.is_active = False
        self.indicator.setText(str(self.step_number))
        self.indicator.setStyleSheet(
            """
            QLabel {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                border: 2px solid #45a049;
            }
        """
        )
        self.title_label.setStyleSheet("color: #999999;")
        self.desc_label.setStyleSheet("color: #666666;")
        self.action_button.setEnabled(True)
        self.action_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )

    def set_active(self, status_text: str = "In Progress..."):
        """Step is currently being executed."""
        self.is_completed = False
        self.is_active = True
        self.indicator.setText("⏳")
        self.indicator.setStyleSheet(
            """
            QLabel {
                background-color: #2196F3;
                color: white;
                border-radius: 20px;
                border: 2px solid #1976D2;
            }
        """
        )
        self.title_label.setStyleSheet("color: #2196F3;")
        self.desc_label.setStyleSheet("color: #2196F3;")
        self.action_button.setText(status_text)
        self.action_button.setEnabled(False)
        self.action_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
            }
        """
        )

    def set_completed(self):
        """Step has been successfully completed."""
        self.is_completed = True
        self.is_active = False
        self.indicator.setText("✓")
        self.indicator.setStyleSheet(
            """
            QLabel {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                border: 2px solid #45a049;
            }
        """
        )
        self.title_label.setStyleSheet("color: #4CAF50;")
        self.desc_label.setStyleSheet("color: #666666;")
        self.action_button.setText("Done")
        self.action_button.setEnabled(False)
        self.action_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
        """
        )


class MainWindow(QWidget):
    """
    Main application window with step-by-step workflow.

    Guides the user through: benchmarking → calibration → data collection → export
    """

    # Signals for user actions
    benchmark_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    collection_requested = pyqtSignal()
    stop_collection_requested = pyqtSignal()
    export_requested = pyqtSignal()
    restart_requested = pyqtSignal()
    help_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.collection_start_time = None
        self.participant_name = None
        self._setup_ui()
        self._ask_participant_name()

    def _ask_participant_name(self):
        """Ask for participant name at startup."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Participant Name",
            "Please enter your name or ID:",
            text="Participant",
        )

        if ok and name.strip():
            self.participant_name = name.strip()
        else:
            self.participant_name = "Anonymous"

        print(f"Participant: {self.participant_name}")

    def _setup_ui(self):
        self.setWindowTitle("Gaze Data Collection Tool")
        self.setMinimumSize(700, 600)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("Gaze Data Collection")
        header_font = QFont()
        header_font.setPointSize(24)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Subtitle
        subtitle = QLabel(
            "Follow these simple steps to collect your gaze tracking data"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666666;")
        main_layout.addWidget(subtitle)

        main_layout.addSpacing(20)

        # Step 1: Benchmarking
        self.step1 = StepWidget(
            1,
            "System Benchmark",
            "Test your computer's performance with the gaze model",
        )
        self.step1.clicked.connect(self.benchmark_requested.emit)
        main_layout.addWidget(self.step1)

        # Step 2: Calibration
        self.step2 = StepWidget(
            2,
            "Eye Calibration",
            "Click on 9 points to calibrate your gaze tracking",
        )
        self.step2.clicked.connect(self.calibration_requested.emit)
        main_layout.addWidget(self.step2)

        # Step 3: Data Collection (starts automatically after calibration)
        self.step3 = StepWidget(
            3,
            "Data Collection",
            "Recording will start automatically after calibration",
        )
        # No click handler - collection starts automatically
        main_layout.addWidget(self.step3)

        # Timer display for collection
        self.timer_label = QLabel("Time: 0:00")
        timer_font = QFont()
        timer_font.setPointSize(14)
        timer_font.setBold(True)
        self.timer_label.setFont(timer_font)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("color: #2196F3; margin: 5px;")
        self.timer_label.hide()
        main_layout.addWidget(self.timer_label)

        # Stop collection button
        self.stop_button = QPushButton("⏹ Stop Collection")
        self.stop_button.setFixedHeight(40)
        stop_font = QFont()
        stop_font.setPointSize(12)
        stop_font.setBold(True)
        self.stop_button.setFont(stop_font)
        self.stop_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """
        )
        self.stop_button.clicked.connect(self.stop_collection_requested.emit)
        self.stop_button.hide()
        main_layout.addWidget(self.stop_button)

        # Step 4: Upload
        self.step4 = StepWidget(
            4, "Upload Data", "Upload your collected data to the server"
        )
        self.step4.clicked.connect(self.export_requested.emit)
        main_layout.addWidget(self.step4)

        main_layout.addStretch()

        # Bottom buttons (Help and Restart)
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        self.help_button = QPushButton("❓ Help")
        self.help_button.setFixedSize(100, 35)
        self.help_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )
        self.help_button.clicked.connect(self.help_requested.emit)

        self.restart_button = QPushButton("🔄 Restart")
        self.restart_button.setFixedSize(100, 35)
        self.restart_button.setStyleSheet(
            """
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """
        )
        self.restart_button.clicked.connect(self._on_restart_clicked)

        bottom_layout.addWidget(self.help_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.restart_button)

        main_layout.addLayout(bottom_layout)

        # Initialize with Step 1 ready
        self.step1.set_ready()

    def _on_restart_clicked(self):
        """Confirm restart with the user."""
        reply = QMessageBox.question(
            self,
            "Restart Session",
            "Are you sure you want to restart?\n\nThis will discard any current progress.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.restart_requested.emit()
            self.reset_ui()

    def reset_ui(self):
        """Reset all steps to initial state."""
        self.step1.set_ready()
        self.step2.set_pending()
        self.step3.set_pending()
        self.step4.set_pending()
        self.timer_label.hide()
        self.stop_button.hide()
        self.collection_start_time = None

    # --- Step State Management Methods ---

    @pyqtSlot()
    def on_benchmark_started(self):
        """Called when benchmarking begins."""
        self.step1.set_active("Testing...")

    @pyqtSlot()
    def on_benchmark_completed(self):
        """Called when benchmarking is complete."""
        self.step1.set_completed()
        self.step2.set_ready()

    @pyqtSlot()
    def on_calibration_started(self):
        """Called when calibration begins."""
        self.step2.set_active("Calibrating...")

    @pyqtSlot()
    def on_calibration_completed(self):
        """Called when calibration is complete."""
        self.step2.set_completed()
        self.step3.set_ready()

    @pyqtSlot()
    def on_collection_started(self):
        """Called when data collection begins."""
        self.step3.set_active("Recording...")
        self.timer_label.show()
        self.stop_button.show()
        import time

        self.collection_start_time = time.time()

    @pyqtSlot(int)
    def on_collection_time_update(self, elapsed_seconds: int):
        """Update the timer display during collection."""
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        self.timer_label.setText(f"Time: {minutes}:{seconds:02d}")

    @pyqtSlot()
    def on_collection_stopped(self):
        """Called when data collection is stopped."""
        self.step3.set_completed()
        self.timer_label.hide()
        self.stop_button.hide()
        self.step4.set_ready()

    @pyqtSlot()
    def on_export_started(self):
        """Called when export begins."""
        self.step4.set_active("Exporting...")

    @pyqtSlot()
    def on_export_completed(self):
        """Called when export is complete."""
        self.step4.set_completed()
        self._show_completion_dialog()

    def _show_completion_dialog(self):
        """Show final completion dialog and exit."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Thank You!")
        msg.setText("🎉 Data collection complete!")
        msg.setInformativeText(
            "Your gaze data has been successfully collected and uploaded.\n\n"
            "Thank you for your participation!"
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.addButton("Exit", QMessageBox.ButtonRole.AcceptRole)

        msg.exec()
        self.close()

    @pyqtSlot()
    def show_help_dialog(self):
        """Show help information dialog."""
        help_text = """
<h2>📖 Gaze Data Collection Tool - Help</h2>

<h3>What is this tool?</h3>
<p><b>⚠️ RESEARCH PURPOSE ONLY</b></p>
<p>This application collects eye gaze tracking data <b>for research purposes only</b>.
The collected data will be used exclusively for academic research and will not be used for any other purpose.
It records where you look on the screen while using your computer.</p>

<h3>How to use:</h3>

<p><b>Step 1: System Benchmark</b><br>
Tests how fast your computer can process gaze tracking.
This takes about 5 seconds. Just click "Start" and wait.</p>

<p><b>Step 2: Eye Calibration</b><br>
You'll see 9 dots appear on your screen one by one.
<br>• Click on each dot while looking at the dot.
<br>• After clicking, rest, then press the SPACE bar to continue.
<br>• Keep head still during calibration.</p>

<p><b>Step 3: Data Collection</b><br>
The app will record your gaze and your screen while you use your computer normally.
<br>• Use your computer as you normally would
<br>• Try to keep your head in a similar position as during calibration
<br>• Random dots may appear occasionally - click on them when they do
<br>• When you're done, click "Stop Collection"</p>

<p><b>Step 4: Upload Data</b><br>
Uploads all your collected data to the research server.</p>

<h3>Tips:</h3>
<p>• Keep your head relatively still throughout
<br>• Good lighting helps the camera see your face better
<br>• Collect data for at least 2-3 minutes for best results
<br>• If you make a mistake, use the "Restart" button</p>

<h3>What is Recorded:</h3>
<p>This tool records:<br>
• Video of your face (for gaze estimation)
<br>• Your screen activity (for gaze research context)
<br>• Where you click on the screen
<br>• Eye tracking data (angles and positions)</p>

<h3>Privacy & Research Use:</h3>
<p><b>Important:</b> All collected data is for research purposes only.
<br>• Your data will only be used for academic research
<br>• Data will not be shared for commercial purposes
<br>• Data will not be used for any surveillance or non-research activities</p>

<p>By participating, you consent to the collection of your face video and screen recording for research purposes.</p>
        """

        # Create custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Help")
        dialog.setGeometry(100, 100, 700, 600)  # width, height

        layout = QVBoxLayout(dialog)

        # Text display
        text_display = QTextEdit(dialog)
        text_display.setHtml(help_text)
        text_display.setReadOnly(True)
        layout.addWidget(text_display)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec()

    @pyqtSlot(str)
    def show_error(self, message: str):
        """Show an error message to the user."""
        QMessageBox.critical(self, "Error", message)

    @pyqtSlot(str, str)
    def show_warning(self, title: str, message: str):
        """Show a warning message to the user."""
        QMessageBox.warning(self, title, message)

    @pyqtSlot(str, str)
    def show_info(self, title: str, message: str):
        """Show an informational message to the user."""
        QMessageBox.information(self, title, message)
