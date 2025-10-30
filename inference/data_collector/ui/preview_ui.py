"""
Simple UI Preview - Test the MainWindow appearance

This is a standalone script to preview the UI without running the full application.
Use this to test the visual design and layout.
"""

import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Add parent directory to path
sys.path.insert(0, "/Users/tanerijun/projects/master-research/my-gaze-model/inference")

from data_collector.ui.main_window import MainWindow


def test_ui_states(window):
    """Simulate different states to preview the UI."""

    def step1():
        print("Step 1: Benchmarking started")
        window.on_benchmark_started()

    def step2():
        print("Step 1: Completed")
        window.on_benchmark_completed()

    def step3():
        print("Step 2: Calibration started")
        window.on_calibration_started()

    def step4():
        print("Step 2: Completed")
        window.on_calibration_completed()

    def step5():
        print("Step 3: Collection started")
        window.on_collection_started()
        # Start timer updates
        timer_counter[0] = 0
        timer.start(1000)

    def update_timer():
        timer_counter[0] += 1
        window.on_collection_time_update(timer_counter[0])
        if timer_counter[0] >= 10:  # Stop after 10 seconds
            timer.stop()
            step6()

    def step6():
        print("Step 3: Collection stopped")
        window.on_collection_stopped()

    def step7():
        print("Step 4: Export started")
        window.on_export_started()

    def step8():
        print("Step 4: Export completed")
        window.on_export_completed()

    # Schedule state changes
    timer_counter = [0]
    timer = QTimer()
    timer.timeout.connect(update_timer)

    # Uncomment these to see automatic progression through states:
    # QTimer.singleShot(2000, step1)   # After 2s: Start benchmark
    # QTimer.singleShot(4000, step2)   # After 4s: Complete benchmark
    # QTimer.singleShot(6000, step3)   # After 6s: Start calibration
    # QTimer.singleShot(8000, step4)   # After 8s: Complete calibration
    # QTimer.singleShot(10000, step5)  # After 10s: Start collection
    # QTimer.singleShot(22000, step7)  # After 22s: Start export
    # QTimer.singleShot(24000, step8)  # After 24s: Complete export


def main():
    app = QApplication(sys.argv)

    window = MainWindow()

    # Connect signals for testing
    window.benchmark_requested.connect(lambda: print("🔔 Benchmark requested"))
    window.calibration_requested.connect(lambda: print("🔔 Calibration requested"))
    window.collection_requested.connect(lambda: print("🔔 Collection requested"))
    window.stop_collection_requested.connect(
        lambda: print("🔔 Stop collection requested")
    )
    window.export_requested.connect(lambda: print("🔔 Export requested"))
    window.restart_requested.connect(lambda: print("🔔 Restart requested"))
    window.help_requested.connect(lambda: window.show_help_dialog())

    # Test state transitions (optional)
    test_ui_states(window)

    window.show()

    print("\n" + "=" * 60)
    print("UI Preview Running")
    print("=" * 60)
    print("\nTry clicking the buttons to see what signals are emitted.")
    print("- Click 'Help' to see the help dialog")
    print("- Click 'Restart' to see the confirmation dialog")
    print("- Click step buttons when they're enabled")
    print("\nClose the window to exit.\n")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
