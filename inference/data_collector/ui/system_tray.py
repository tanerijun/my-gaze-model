from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from data_collector import config


class SystemTray(QObject):
    """Manages the system tray icon and its menu."""

    start_collection_requested = pyqtSignal()
    stop_collection_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("System tray is not available on this system.")

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(str(config.ICON_PATH)))
        self.tray_icon.setToolTip("Gaze Tracker")

        self.menu = QMenu()
        self._create_actions()
        self.tray_icon.setContextMenu(self.menu)

        self.tray_icon.show()
        print("System tray icon initialized.")

    def _create_actions(self):
        """Creates the menu actions and connects them."""
        self.start_action = QAction("🟢 Start Collection")
        self.start_action.triggered.connect(self.start_collection_requested.emit)
        self.menu.addAction(self.start_action)

        self.menu.addSeparator()

        self.about_action = QAction("ℹ️ About")
        # self.about_action.triggered.connect(self.show_about_dialog) # Placeholder
        self.menu.addAction(self.about_action)

        self.quit_action = QAction("❌ Quit")
        self.quit_action.triggered.connect(QApplication.quit)
        self.menu.addAction(self.quit_action)

    def set_menu_state(self, is_collecting: bool):
        """Updates the menu text and enabled state based on app status."""
        if is_collecting:
            self.start_action.setText("🔴 Stop Collection")
            # Disconnect start, connect stop
            try:
                self.start_action.triggered.disconnect()
            except TypeError:
                pass
            self.start_action.triggered.connect(self.stop_collection_requested.emit)
        else:
            self.start_action.setText("🟢 Start Collection")
            # Disconnect stop, connect start
            try:
                self.start_action.triggered.disconnect()
            except TypeError:
                pass
            self.start_action.triggered.connect(self.start_collection_requested.emit)
