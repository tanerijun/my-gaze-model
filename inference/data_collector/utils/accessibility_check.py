import platform
import sys

from pynput import mouse
from PyQt6.QtWidgets import QApplication, QMessageBox


def check_accessibility_permission() -> bool:
    """
    Check if the application has accessibility permissions to monitor input events.

    On macOS, pynput requires accessibility permissions to monitor mouse/keyboard events.
    This function checks the listener's IS_TRUSTED property.

    Returns:
        bool: True if permissions appear to be granted, False otherwise.
    """
    # Only relevant for macOS
    if platform.system() != "Darwin":
        return True

    try:
        listener = mouse.Listener(on_click=lambda x, y, button, pressed: None)
        is_trusted = listener.IS_TRUSTED  # type:ignore
        return is_trusted

    except Exception as e:
        print(f"Error checking accessibility permissions: {e}", file=sys.stderr)
        return False


def log_accessibility_status():
    """
    Check and log the accessibility permission status to console.
    Exits the application if permissions are not granted on macOS.
    """
    if platform.system() != "Darwin":
        print("Accessibility check: Not required on this platform")
        return

    print("Checking accessibility permissions...")

    if not check_accessibility_permission():
        # Create a temporary QApplication if one doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Show error dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Accessibility Permissions Required")
        msg.setText("Accessibility permissions not granted!")
        msg.setInformativeText(
            "This application requires accessibility permissions to monitor "
            "mouse clicks for gaze data collection.\n\n"
            "Please grant accessibility permissions in:\n"
            "System Settings > Privacy & Security > Accessibility\n\n"
            "Add this application (or your terminal/IDE) to the list and "
            "ensure the checkbox is enabled.\n\n"
            "After granting permissions, restart the application."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        print("Accessibility permissions: Not available")

        sys.exit(1)

    print("Accessibility permissions: OK")
