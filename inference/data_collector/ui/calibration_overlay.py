"""
Calibration Overlay Widget

A frameless, transparent, always-on-top QWidget that displays calibration points,
instructions, and drift warnings. Users interact with this overlay by clicking on
calibration points and using keyboard shortcuts.
"""

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class CalibrationOverlay(QWidget):
    """
    A frameless, transparent overlay that displays calibration points and instructions.

    Signals:
        - point_clicked: Emitted when user clicks on a calibration point. Args: (x, y)
        - cancel_requested: Emitted when user presses ESC.
        - continue_requested: Emitted when user presses Space.

    The overlay should be shown with `showFullScreen()` and can be controlled via:
        - set_calibration_point(x, y): Display a calibration point
        - set_warning_message(message): Show a warning message
        - clear_warning(): Hide the warning
    """

    point_clicked = pyqtSignal(int, int)  # x, y coordinates of the click
    cancel_requested = pyqtSignal()
    continue_requested = pyqtSignal()

    # Configuration constants
    POINT_RADIUS = 12
    POINT_COLOR = QColor(0, 150, 255)  # Bright blue
    POINT_OUTLINE_WIDTH = 2
    TEXT_COLOR = QColor(255, 255, 255)  # White
    WARNING_COLOR = QColor(255, 100, 100)  # Light red for warnings
    SUCCESS_COLOR = QColor(100, 255, 100)  # Light green for success
    INSTRUCTION_TEXT_SIZE = 14
    WARNING_TEXT_SIZE = 16
    CLICK_FEEDBACK_DURATION_MS = 200  # Duration to show click feedback

    def __init__(self, parent=None):
        super().__init__(parent)

        # Window flags: frameless, transparent, always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Current calibration point (screen coordinates)
        self.calibration_point: Optional[tuple[int, int]] = None
        self.last_gaze_result = None

        # Warning message and its color
        self.warning_message: Optional[str] = None
        self.warning_color = self.WARNING_COLOR  # Default to red for warnings

        # Central message (different from warning - for rest instructions)
        self.central_message: Optional[str] = None

        # Click feedback: highlight the point briefly after a click
        self.show_click_feedback = False
        self.click_feedback_timer = QTimer()
        self.click_feedback_timer.setSingleShot(True)
        self.click_feedback_timer.timeout.connect(self._on_click_feedback_timeout)

        # Show/hide instruction text
        self.show_instructions = False  # Start hidden, show only after point disappears

        # Prevent double clicks
        self.click_accepted = False  # Only allow one click per point

        # Pulsating animation
        self.pulsate_phase = 0.0  # 0.0 to 1.0 for pulsation
        self.pulsate_timer = QTimer()
        self.pulsate_timer.timeout.connect(self._update_pulsate)
        self.pulsate_timer.setInterval(16)  # ~60fps

        # Set a semi-transparent dark background for the entire overlay
        self.setStyleSheet("background-color: rgba(0, 0, 0, 20);")

        print("CalibrationOverlay initialized.")

    def set_calibration_point(self, x: int, y: int):
        """
        Set a new calibration point to display.

        Args:
            x, y: Screen coordinates of the calibration point
        """
        self.calibration_point = (x, y)
        self.show_click_feedback = False
        self.warning_message = None
        self.central_message = None  # Clear central message when showing point
        self.click_accepted = False  # Reset click flag for new point
        self.pulsate_phase = 0.0
        self.pulsate_timer.stop()  # Don't start pulsating yet, wait for click
        self.update()

    def clear_calibration_point(self):
        """Hide the calibration point."""
        self.calibration_point = None
        self.show_click_feedback = False
        self.pulsate_timer.stop()  # Stop pulsating
        self.update()

    def _update_pulsate(self):
        """Update pulsation animation phase."""
        self.pulsate_phase = (self.pulsate_phase + 0.02) % 1.0  # Slower: 0.05 -> 0.02
        self.update()

    def set_warning_message(self, message: str, success: bool = False):
        """
        Display a message (warning or success).

        Args:
            message: The message text to display
            success: If True, use success color (green); if False, use warning color (red)
        """
        self.warning_message = message
        self.warning_color = self.SUCCESS_COLOR if success else self.WARNING_COLOR
        self.update()

    def clear_warning(self):
        """Hide the warning message."""
        self.warning_message = None
        self.update()

    def set_central_message(self, message: str):
        """
        Display a central message (for rest instructions).

        Args:
            message: The message text to display in the center
        """
        self.central_message = message
        self.update()

    def clear_central_message(self):
        """Hide the central message."""
        self.central_message = None
        self.update()

    def show_instruction_text(self, show: bool = True):
        """
        Control visibility of the 'Press SPACE to continue' instruction text.

        Args:
            show: If True, show instructions; if False, hide them
        """
        self.show_instructions = show
        self.update()

    def paintEvent(self, a0: "QPaintEvent") -> None:  # type: ignore  # noqa: F821
        """Draws the overlay elements."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw calibration point if set
        if self.calibration_point:
            self._draw_calibration_point(painter)

        # Draw warning message if set
        if self.warning_message:
            self._draw_warning_message(painter)

        # Draw central message if set
        if self.central_message:
            self._draw_central_message(painter)

        # Always draw instructions
        self._draw_instructions(painter)

        painter.end()

    def _draw_calibration_point(self, painter: QPainter):
        """Draws a calibration point with optional click feedback and pulsation."""
        if not self.calibration_point:
            return
        x, y = self.calibration_point

        # Calculate pulsation effect (scale between 0.95 and 1.05 - smaller range)
        import math

        pulsate_scale = 1.0 + 0.05 * math.sin(self.pulsate_phase * 2 * math.pi)
        pulsate_radius = int(self.POINT_RADIUS * pulsate_scale)

        # Outer circle (outline) with pulsation
        outline_color = (
            QColor(100, 200, 255) if self.show_click_feedback else self.POINT_COLOR
        )
        painter.setPen(
            QPen(outline_color, self.POINT_OUTLINE_WIDTH, Qt.PenStyle.SolidLine)
        )

        # Add glow effect during pulsation
        alpha = int(50 + 30 * math.sin(self.pulsate_phase * 2 * math.pi))
        painter.setBrush(
            QColor(
                outline_color.red(), outline_color.green(), outline_color.blue(), alpha
            )
        )
        painter.drawEllipse(QPoint(x, y), pulsate_radius, pulsate_radius)

        # Inner filled circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(outline_color)
        painter.drawEllipse(
            QPoint(x, y), self.POINT_RADIUS // 2, self.POINT_RADIUS // 2
        )

        # Draw center dot for precision
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPoint(x, y), 2, 2)

    def _draw_warning_message(self, painter: QPainter):
        """Draws the warning/success message in the center of the screen."""
        painter.setPen(self.warning_color)
        font = QFont("Arial", self.WARNING_TEXT_SIZE, QFont.Weight.Bold)
        painter.setFont(font)

        # Draw in the center of the screen
        screen_rect = self.rect()
        painter.drawText(
            screen_rect,
            Qt.AlignmentFlag.AlignCenter,
            self.warning_message,
        )

    def _draw_central_message(self, painter: QPainter):
        """Draws the central message (rest instructions) in the center of the screen."""
        painter.setPen(self.TEXT_COLOR)
        font = QFont("Arial", 18, QFont.Weight.Normal)
        painter.setFont(font)

        # Draw in the center of the screen
        screen_rect = self.rect()
        painter.drawText(
            screen_rect,
            Qt.AlignmentFlag.AlignCenter,
            self.central_message,
        )

    def _draw_instructions(self, painter: QPainter):
        """Draws keyboard instruction text at the bottom of the screen (only if enabled)."""
        if not self.show_instructions:
            return

        painter.setPen(self.TEXT_COLOR)
        font = QFont("Arial", self.INSTRUCTION_TEXT_SIZE)
        painter.setFont(font)

        instructions = "Space: Continue  |  ESC: Cancel"
        screen_rect = self.rect()

        # Draw at the bottom, with some padding
        rect = QRect(0, screen_rect.height() - 50, screen_rect.width(), 40)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            instructions,
        )

    def mousePressEvent(self, a0: QMouseEvent) -> None:  # type: ignore
        """Handles mouse clicks on the overlay."""
        # Prevent double clicks - only accept first click
        if (
            self.calibration_point
            and not self.warning_message
            and not self.click_accepted
        ):
            # Only register clicks if a calibration point is visible and no warning is active
            x = a0.position().x()
            y = a0.position().y()

            # Check if the click is close enough to the calibration point
            cal_x, cal_y = self.calibration_point
            distance = ((x - cal_x) ** 2 + (y - cal_y) ** 2) ** 0.5

            # Allow a click radius of 40 pixels
            if distance <= 40:
                # Mark that we've accepted a click for this point
                self.click_accepted = True

                # Start pulsating animation for the 500ms wait period
                self.pulsate_phase = 0.0
                self.pulsate_timer.start()

                # Emit the click signal with screen coordinates
                self.point_clicked.emit(int(x), int(y))

                # Show click feedback
                self.show_click_feedback = True
                self.update()
                self.click_feedback_timer.start(self.CLICK_FEEDBACK_DURATION_MS)
            else:
                # Click was too far from the point, ignore it
                pass
        a0.accept()

    def keyPressEvent(self, a0: QKeyEvent) -> None:  # type: ignore
        """Handles keyboard input."""
        if a0.key() == Qt.Key.Key_Space and not a0.isAutoRepeat():
            self.continue_requested.emit()
            a0.accept()
        elif a0.key() == Qt.Key.Key_Escape and not a0.isAutoRepeat():
            self.cancel_requested.emit()
            a0.accept()
        else:
            super().keyPressEvent(a0)

    def _on_click_feedback_timeout(self):
        """Callback when click feedback timer expires."""
        self.show_click_feedback = False
        self.update()
