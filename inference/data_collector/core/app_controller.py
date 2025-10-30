import datetime
from enum import Enum, auto
from queue import Queue

import numpy as np
from PyQt6.QtCore import (
    Q_ARG,
    QMetaObject,
    QObject,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import QApplication

from data_collector import config
from data_collector.core.data_manager import DataManager
from data_collector.core.workers import CameraWorker, InferenceWorker, StorageWorker
from data_collector.ui.calibration_overlay import CalibrationOverlay
from data_collector.utils.system_info import get_system_info


class AppState(Enum):
    IDLE = auto()
    BENCHMARKING = auto()
    READY_TO_CALIBRATE = auto()
    CALIBRATING = auto()
    COLLECTING = auto()
    PAUSED_BY_DRIFT = auto()


def _make_json_serializable(obj):  # type: ignore
    """
    Convert an object to a JSON-serializable format.
    Handles numpy arrays, custom objects with __dict__, and nested structures.
    """
    if obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        # Convert custom objects to dict
        return _make_json_serializable(obj.__dict__)
    else:
        # For any other type, try to convert to string
        return str(obj)


def _filter_gaze_result(gaze_result):
    """
    Filter gaze result to keep only necessary fields for data collection.
    Removes large/unnecessary data like head_pose_matrix, mediapipe_landmarks, blaze_keypoints.
    """
    if not gaze_result:
        return None

    # Only keep the fields we care about
    filtered = {}

    # Keep gaze angles (pitch, yaw, roll)
    if "gaze" in gaze_result:
        filtered["gaze"] = {
            "pitch": gaze_result["gaze"].get("pitch"),
            "yaw": gaze_result["gaze"].get("yaw"),
            "roll": gaze_result["gaze"].get("roll"),
        }

    # Keep eye-related measurements
    if "eye_distance" in gaze_result:
        filtered["eye_distance"] = gaze_result["eye_distance"]

    if "eye_center" in gaze_result:
        filtered["eye_center"] = gaze_result["eye_center"]

    # Keep face bbox for reference
    if "face_bbox" in gaze_result:
        filtered["face_bbox"] = gaze_result["face_bbox"]

    # Convert to JSON-serializable format
    return _make_json_serializable(filtered)


class AppController(QObject):
    """
    Orchestrates all workers and threads for the application.
    """

    # Signal to command the storage worker to start
    start_recording_signal = pyqtSignal(str, int, int, int)
    # New signal to update the UI text
    state_changed = pyqtSignal(AppState)

    def __init__(self):
        super().__init__()
        self.state = AppState.IDLE
        self.session_metadata = {}

        self.benchmark_frame_count = 0
        self.benchmark_start_time = 0.0

        self.video_queue = Queue(maxsize=120)  # Buffer ~2s of video at 60fps
        self.inference_queue = Queue(maxsize=5)  # Small buffer, we only need the latest

        # --- Setup Data Manager ---
        self.data_manager = DataManager(config.DATA_OUTPUT_DIR)

        # --- Setup Calibration Overlay ---
        self.overlay = CalibrationOverlay()
        self.overlay.point_clicked.connect(self.on_calibration_point_clicked)
        self.overlay.cancel_requested.connect(self.on_calibration_cancelled)
        self.overlay.continue_requested.connect(self.on_calibration_continue)

        # --- Setup Camera Thread ---
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(
            config.CAMERA_ID,
            config.DESIRED_CAMERA_RESOLUTION,
            self.video_queue,
            self.inference_queue,
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.camera_error.connect(self.on_camera_error)

        # --- Setup Inference Worker & Thread ---
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker(
            str(config.GAZE_MODEL_PATH), config.PIPELINE_CONFIG, self.inference_queue
        )
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.started.connect(self.inference_worker.setup)
        self.inference_worker.result_ready.connect(self.on_new_result)
        self.inference_worker.benchmark_finished.connect(self.on_benchmark_finished)

        # --- Setup Storage Worker & Thread ---
        self.storage_thread = QThread()
        self.storage_worker = StorageWorker(self.video_queue)
        self.storage_worker.moveToThread(self.storage_thread)
        self.start_recording_signal.connect(self.storage_worker.setup_and_run)

        print("AppController initialized.")

    def _set_state(self, new_state: AppState):
        if self.state == new_state:
            return
        self.state = new_state
        print(f"State changed to: {new_state.name}")
        self.state_changed.emit(new_state)

    @pyqtSlot()
    def start_session(self):
        if self.state != AppState.IDLE:
            return

        print("\n--- Starting Session: Stage 1 - Benchmarking ---")
        self._set_state(AppState.BENCHMARKING)
        self.session_metadata = {}
        self.session_metadata["system_info"] = get_system_info()
        screen = QApplication.primaryScreen()
        if not screen:
            raise ValueError("screen is None")
        self.session_metadata["screen_size"] = {
            "width": screen.size().width(),
            "height": screen.size().height(),
        }

        # Recreate threads if they were stopped before
        if not self.camera_thread.isRunning():
            self.camera_thread = QThread()
            self.camera_worker = CameraWorker(
                config.CAMERA_ID,
                config.DESIRED_CAMERA_RESOLUTION,
                self.video_queue,
                self.inference_queue,
            )
            self.camera_worker.moveToThread(self.camera_thread)
            self.camera_thread.started.connect(self.camera_worker.run)
            self.camera_worker.camera_error.connect(self.on_camera_error)

        if not self.inference_thread.isRunning():
            self.inference_thread = QThread()
            self.inference_worker = InferenceWorker(
                str(config.GAZE_MODEL_PATH),
                config.PIPELINE_CONFIG,
                self.inference_queue,
            )
            self.inference_worker.moveToThread(self.inference_thread)
            self.inference_thread.started.connect(self.inference_worker.setup)
            self.inference_worker.result_ready.connect(self.on_new_result)
            self.inference_worker.benchmark_finished.connect(self.on_benchmark_finished)

        # prevents video queue from blocking
        self.camera_worker.set_video_recording(False)

        # signals needed ONLY for the benchmark process
        self.camera_worker.camera_started.connect(self.on_camera_ready_for_benchmark)
        self.camera_worker.frame_ready.connect(self.on_receive_benchmark_frame)

        self.camera_thread.start()
        self.inference_thread.start()

    def start_data_collection(self):
        """This method will be called when the user clicks 'Start Calibration'."""
        if self.state != AppState.READY_TO_CALIBRATE:
            return

        print("\n--- Starting Session: Stage 2 - Calibration ---")
        self._set_state(AppState.CALIBRATING)

        # Initialize data manager with session metadata
        session_id = self.data_manager.start_session(self.session_metadata)
        print(f"Session ID: {session_id}")

        # Initialize calibration state tracking
        self.calibration_points_collected = []  # List of (target_x, target_y, actual_x, actual_y, head_pose)
        self.calibration_grid = self._generate_calibration_grid()
        self.calibration_current_index = 0
        self._calibration_started = False
        self._waiting_for_space = False  # Flag to require Space before next point

        # Track video start time for timestamp calculations
        self.video_start_time = datetime.datetime.now()

        # Enable video recording
        self.camera_worker.set_video_recording(True)

        # Start threads for the calibration phase
        self.camera_worker.camera_started.connect(self.on_camera_ready_for_recording)
        self.storage_thread.start()
        self.inference_thread.started.connect(self.inference_worker.run)
        self.inference_thread.start()
        self.camera_thread.start()

        # After a brief delay, show the first calibration point
        # We'll do this after on_new_result starts receiving pose data
        print("Waiting for first frame...")

    def _generate_calibration_grid(self) -> list:
        """Generate a 3x3 grid of calibration points with aspect-ratio-aware margins."""
        screen = QApplication.primaryScreen()
        if not screen:
            raise ValueError("screen is None")
        width = screen.size().width()
        height = screen.size().height()

        # Use different margins for horizontal and vertical
        margin_horizontal = 0.04  # 4% margin on left/right
        margin_vertical = 0.08  # 8% margin on top/bottom

        print(f"Screen size: {width}x{height}")
        print(f"Margins: horizontal={margin_horizontal}, vertical={margin_vertical}")

        points = []
        for i in range(3):  # rows (top to bottom)
            for j in range(3):  # columns (left to right)
                # Calculate position:
                # j=0 -> margin_horizontal, j=1 -> 0.5, j=2 -> 1-margin_horizontal
                # i=0 -> margin_vertical, i=1 -> 0.5, i=2 -> 1-margin_vertical
                x_ratio = (
                    margin_horizontal
                    if j == 0
                    else (0.5 if j == 1 else 1 - margin_horizontal)
                )
                y_ratio = (
                    margin_vertical
                    if i == 0
                    else (0.5 if i == 1 else 1 - margin_vertical)
                )

                x = int(width * x_ratio)
                y = int(height * y_ratio)
                print(f"Point [{i},{j}]: ({x}, {y}) = ({x_ratio:.2%}, {y_ratio:.2%})")
                points.append((x, y))
        return points

    @pyqtSlot()
    def stop_session(self):
        if self.state == AppState.IDLE:
            return
        print("Controller: Stopping all active workers...")

        # Close overlay if it's visible
        if self.overlay.isVisible():
            self.overlay.close()
            QApplication.processEvents()

        # Signal all workers to stop
        self.camera_worker.stop()
        self.inference_worker.stop()
        self.storage_worker.stop()

        # Quit the threads and wait for them to finish
        for thread in [self.camera_thread, self.inference_thread, self.storage_thread]:
            if thread.isRunning():
                thread.quit()
                thread.wait()

        # Clear queues in case there's leftover data
        while not self.video_queue.empty():
            self.video_queue.get_nowait()
        while not self.inference_queue.empty():
            self.inference_queue.get_nowait()

        # Save session data to disk
        json_path = self.data_manager.save_to_disk()
        if json_path:
            print(f"Session data saved to: {json_path}")

        self._set_state(AppState.IDLE)
        print("Controller: All processes stopped. Ready for new session.")

    # --- Benchmark-Specific Slots ---

    @pyqtSlot(int, int)
    def on_camera_ready_for_benchmark(self, width: int, height: int):
        print("Camera is ready. Capturing a single frame for benchmark.")
        self.session_metadata["camera_resolution"] = {"width": width, "height": height}

        # Disconnect to prevent this from running again during normal collection
        self.camera_worker.camera_started.disconnect(self.on_camera_ready_for_benchmark)

    @pyqtSlot(np.ndarray)
    def on_receive_benchmark_frame(self, frame: np.ndarray):
        print("Single frame captured. Stopping camera and starting benchmark.")

        # We only need one frame, so disconnect immediately.
        self.camera_worker.frame_ready.disconnect(self.on_receive_benchmark_frame)

        # Stop the camera; its job is done for the benchmark
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait()

        # invoke the benchmark method on the inference worker's thread
        QMetaObject.invokeMethod(
            self.inference_worker,
            "run_benchmark",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(np.ndarray, frame.copy()),
            Q_ARG(int, config.BENCHMARK_FRAMES),
        )

    @pyqtSlot(float)
    def on_benchmark_finished(self, fps: float):
        self.session_metadata["performance"] = {"inference_fps": round(fps, 2)}

        print("\n--- Benchmarking Complete ---")
        self._print_session_summary()

        # Stop the inference thread. It will be restarted fresh for data collection.
        self.inference_thread.quit()
        self.inference_thread.wait()

        self.camera_worker.set_video_recording(True)

        self._set_state(AppState.READY_TO_CALIBRATE)
        print("\nReady to start calibration.")

    # --- Data Collection Slots ---

    @pyqtSlot(int, int)
    def on_camera_ready_for_recording(self, width: int, height: int):
        # Use the session_id from data_manager to ensure consistency
        session_id = self.data_manager.session_id
        video_filename = f"session_{session_id}.mp4"
        session_dir = config.DATA_OUTPUT_DIR / f"session_{session_id}"
        output_path = str(session_dir / video_filename)

        # Store video filename in data manager
        self.data_manager.set_video_filename(video_filename)

        self.start_recording_signal.emit(output_path, width, height, config.VIDEO_FPS)
        self.camera_worker.camera_started.disconnect(self.on_camera_ready_for_recording)

    @pyqtSlot(list)
    def on_new_result(self, results: list):
        """Handles new gaze results during calibration and collection phases."""
        if not results:
            return

        # Store the most recent result for later use
        self.last_gaze_result = results[0]

        if self.state == AppState.CALIBRATING:
            # Show the next calibration point if we haven't started yet
            if not self._calibration_started:
                self._calibration_started = True
                self._show_next_calibration_point()
        elif self.state == AppState.COLLECTING:
            gaze = results[0]["gaze"]

            # Check for head pose drift
            is_drifted, drift_message = self._check_head_pose_drift(results[0])

            if is_drifted:
                # Pause collection and show warning
                self._set_state(AppState.PAUSED_BY_DRIFT)
                # Clear all previous overlay messages first
                self.overlay.clear_calibration_point()
                self.overlay.clear_central_message()
                self.overlay.show_instruction_text(False)
                # Show warning message
                self.overlay.set_warning_message(
                    drift_message + "\n\nPlease return to your original position"
                )
                self.overlay.show_as_overlay()
                print(f"\n{drift_message} - Data collection paused")

            print(
                f"Gaze Result: Pitch={gaze['pitch']:.1f}, Yaw={gaze['yaw']:.1f}",
                end="\r",
            )
        elif self.state == AppState.PAUSED_BY_DRIFT:
            # Check if user has returned to acceptable pose
            is_drifted, _ = self._check_head_pose_drift(results[0])

            if not is_drifted:
                # Resume collection
                print("\n✓ Head pose restored - Resuming data collection")
                self.overlay.clear_warning()
                self.overlay.close()
                self._set_state(AppState.COLLECTING)

    @pyqtSlot(int, int)
    def on_calibration_point_clicked(self, x: int, y: int):
        """Handles a click on the calibration overlay during calibration."""
        if self.state != AppState.CALIBRATING:
            return

        print(f"\nCalibration point clicked at ({x}, {y})")

        # Get the current target point
        target_x, target_y = self.calibration_grid[self.calibration_current_index]

        # Get the most recent gaze result, filter unnecessary fields, and make it JSON-serializable
        gaze_result = getattr(self, "last_gaze_result", None)
        gaze_result_serializable = _filter_gaze_result(gaze_result)

        # Calculate video timestamp (time since video started)
        video_timestamp = (
            datetime.datetime.now() - self.video_start_time
        ).total_seconds()

        # Record the collected point for local tracking
        self.calibration_points_collected.append(
            {
                "target": (target_x, target_y),
                "click": (x, y),
                "gaze_result": gaze_result,  # Keep original for drift calculation
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        # Save to data manager (use serializable version)
        self.data_manager.add_calibration_point(
            target_x=target_x,
            target_y=target_y,
            click_x=x,
            click_y=y,
            gaze_result=gaze_result_serializable,  # type: ignore
            video_timestamp=video_timestamp,
        )

        print(f"Point {self.calibration_current_index + 1}/9 collected")
        print("Press SPACE to continue to next point...")

        # Move to next calibration point
        self.calibration_current_index += 1

        # Set flag to require Space before showing next point
        self._waiting_for_space = True

        # After 500ms (eye fixation time), the overlay will show the point is done
        # but we still wait for Space
        QTimer.singleShot(500, self._on_fixation_complete)

    def _show_next_calibration_point(self):
        """Displays the next calibration point on the overlay."""
        if self.calibration_current_index >= len(self.calibration_grid):
            return

        # Hide the instruction text and central message
        self.overlay.show_instruction_text(False)
        self.overlay.clear_central_message()

        target_x, target_y = self.calibration_grid[self.calibration_current_index]
        self.overlay.set_calibration_point(target_x, target_y)
        self.overlay.show_as_overlay()
        print(f"Showing calibration point {self.calibration_current_index + 1}/9")

    def _on_fixation_complete(self):
        """Called after 500ms of fixation time on a calibration point."""
        # The point has been displayed for 500ms (eye fixation time)
        # Now hide the point and show the instruction text
        self.overlay.clear_calibration_point()  # Hide the point
        self.overlay.show_instruction_text(True)  # Show "Press SPACE to continue"

        # Show different central message based on whether it's the last point
        if (
            self.calibration_current_index >= 8
        ):  # Last point (0-indexed, so 8 is the 9th point)
            self.overlay.set_central_message(
                "CALIBRATION COMPLETED\nPress SPACE to continue"
            )
        else:
            self.overlay.set_central_message("Rest your eyes\nPress SPACE to continue")

        print("Point disappeared. Ready for next point (press SPACE).")

    def _transition_to_collection(self):
        """Transitions from calibration to data collection phase."""
        self.overlay.close()
        QApplication.processEvents()  # Force process pending events
        self._set_state(AppState.COLLECTING)
        print("Transitioning to data collection...")

    def _finish_calibration(self):
        """Completes the calibration phase."""
        print("\n--- Calibration Complete ---")
        print(f"Collected {len(self.calibration_points_collected)} calibration points")

        # Calculate baseline head pose by averaging the collected data
        if self.calibration_points_collected:
            gaze_results = [
                p["gaze_result"]
                for p in self.calibration_points_collected
                if p["gaze_result"]
            ]
            if gaze_results:
                # Average the head pose data
                avg_head_data = self._average_head_poses(gaze_results)
                self.baseline_head_pose = avg_head_data
                # Save baseline to data manager
                self.data_manager.set_baseline_head_pose(avg_head_data)
                print("Baseline head pose calculated and saved")

        # Immediately close the overlay and force update
        self.overlay.close()
        QApplication.processEvents()  # Force process pending events

        # Transition to collecting state
        self._transition_to_collection()

    def _average_head_poses(self, gaze_results: list) -> dict:
        """Average multiple gaze results to get a baseline head pose."""
        avg_pose = {}
        if gaze_results:
            # Extract only the features we care about for drift detection
            rolls = [
                r["gaze"]["roll"]
                for r in gaze_results
                if "gaze" in r and "roll" in r["gaze"]
            ]
            eye_distances = [
                r["eye_distance"] for r in gaze_results if "eye_distance" in r
            ]
            eye_centers_x = [
                r["eye_center"][0]
                for r in gaze_results
                if "eye_center" in r and len(r["eye_center"]) >= 2
            ]
            eye_centers_y = [
                r["eye_center"][1]
                for r in gaze_results
                if "eye_center" in r and len(r["eye_center"]) >= 2
            ]

            # Calculate averages
            avg_pose["roll"] = sum(rolls) / len(rolls) if rolls else 0.0
            avg_pose["eye_distance"] = (
                sum(eye_distances) / len(eye_distances) if eye_distances else 0.0
            )
            avg_pose["eye_center_x"] = (
                sum(eye_centers_x) / len(eye_centers_x) if eye_centers_x else 0.0
            )
            avg_pose["eye_center_y"] = (
                sum(eye_centers_y) / len(eye_centers_y) if eye_centers_y else 0.0
            )

            # Store metadata about the baseline calculation
            avg_pose["num_samples"] = len(gaze_results)
        return avg_pose

    def _check_head_pose_drift(self, current_result: dict) -> tuple[bool, str]:
        """
        Check if the current head pose has drifted too far from baseline.

        Args:
            current_result: Current gaze result dictionary

        Returns:
            (is_drifted, message): Tuple of boolean and warning message
        """
        if not hasattr(self, "baseline_head_pose") or not self.baseline_head_pose:
            return False, ""

        baseline = self.baseline_head_pose
        current_gaze = current_result.get("gaze", {})
        current_eye_dist = current_result.get("eye_distance", 0.0)
        current_eye_center = current_result.get("eye_center", [0.0, 0.0])

        # Check roll drift
        roll_diff = abs(current_gaze.get("roll", 0.0) - baseline.get("roll", 0.0))
        if roll_diff > config.DRIFT_THRESHOLDS["roll_degrees"]:
            return True, f"⚠️ Head rolled too much (roll: {roll_diff:.1f}°)"

        # Check eye distance (IPD) change - indicates moving closer/farther
        baseline_eye_dist = baseline.get("eye_distance", 0.0)
        if baseline_eye_dist > 0:
            eye_dist_ratio = (
                abs(current_eye_dist - baseline_eye_dist) / baseline_eye_dist
            )
            if eye_dist_ratio > config.DRIFT_THRESHOLDS["eye_distance_ratio"]:
                direction = (
                    "closer" if current_eye_dist > baseline_eye_dist else "farther"
                )
                return (
                    True,
                    f"⚠️ Moved too {direction} from camera ({eye_dist_ratio * 100:.1f}% change)",
                )

        # Check eye center position drift (lateral/vertical head movement)
        baseline_eye_center_x = baseline.get("eye_center_x", 0.0)
        baseline_eye_center_y = baseline.get("eye_center_y", 0.0)
        current_eye_center_x = (
            current_eye_center[0] if len(current_eye_center) >= 2 else 0.0
        )
        current_eye_center_y = (
            current_eye_center[1] if len(current_eye_center) >= 2 else 0.0
        )

        # Calculate Euclidean distance of eye center movement
        eye_center_shift = (
            (current_eye_center_x - baseline_eye_center_x) ** 2
            + (current_eye_center_y - baseline_eye_center_y) ** 2
        ) ** 0.5

        if eye_center_shift > config.DRIFT_THRESHOLDS["eye_center_shift_pixels"]:
            return (
                True,
                f"⚠️ Head shifted position ({eye_center_shift:.1f} pixels)",
            )

        return False, ""

    @pyqtSlot()
    def on_calibration_cancelled(self):
        """Handles cancellation of calibration."""
        print("Calibration cancelled by user")
        self.stop_session()

    @pyqtSlot()
    def on_calibration_continue(self):
        """Handles the continue (Space) key press during calibration."""
        if self.state != AppState.CALIBRATING or not self._waiting_for_space:
            return

        print("Space pressed. Moving to next point...")
        self._waiting_for_space = False

        if self.calibration_current_index < 9:
            self._show_next_calibration_point()
        else:
            # Calibration complete!
            self._finish_calibration()

    # --- Utility and Cleanup Methods ---

    def _print_session_summary(self):
        """Prints a nicely formatted summary of the collected metadata."""
        session_info = self.session_metadata
        sys_info = session_info.get("system_info", {})

        print("\n================ SESSION METADATA ================")
        print(f"OS:            {sys_info.get('os', 'N/A')}")
        print(f"Python Version:{sys_info.get('python_version', 'N/A')}")
        print(f"Torch Version: {sys_info.get('torch_version', 'N/A')}")

        cpu = sys_info.get("cpu", {})
        if cpu:
            print("-------------------- CPU ---------------------")
            print(f"Brand:         {cpu.get('brand', 'N/A')}")
            print(f"Arch:          {cpu.get('arch', 'N/A')}")
            print(f"Base Speed:    {cpu.get('base_speed_ghz', 0):.2f} GHz")
            print(
                f"Cores:         {cpu.get('physical_cores', 0)} Physical / {cpu.get('total_cores', 0)} Total"
            )
            print(f"Usage:         {cpu.get('current_usage_percent', 0)}%")
            print(
                f"L2/L3 Cache:   {cpu.get('l2_cache_size_kb', 0)} KB / {cpu.get('l3_cache_size_kb', 0)} KB"
            )

        ram = sys_info.get("ram", {})
        if ram:
            print("-------------------- RAM ---------------------")
            print(
                f"Total/Avail:   {ram.get('total_gb', 0):.2f} GB / {ram.get('available_gb', 0):.2f} GB"
            )
            print(f"Usage:         {ram.get('usage_percent', 0)}%")

        # Now, access the top-level keys from the main session dictionary
        screen = session_info.get("screen_size", {})
        if screen:
            print(f"Screen Res:    {screen.get('width', 0)}x{screen.get('height', 0)}")

        cam = session_info.get("camera_resolution", {})
        if cam:
            print(f"Camera Res:    {cam.get('width', 0)}x{cam.get('height', 0)}")

        print("--------------------------------------------------")
        perf = session_info.get("performance", {})
        print(f"INFERENCE FPS: {perf.get('inference_fps', 0.0):.2f} FPS")
        print("==================================================")

    def on_camera_error(self, error_msg: str):
        print(f"FATAL CAMERA ERROR: {error_msg}")
        self.stop_session()

    def cleanup(self):
        print("Performing cleanup...")
        self.stop_session()
