import datetime
from enum import Enum, auto
from queue import Queue

import numpy as np
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from data_collector import config
from data_collector.core.workers import CameraWorker, InferenceWorker, StorageWorker
from data_collector.utils.system_info import get_system_info


class AppState(Enum):
    IDLE = auto()
    BENCHMARKING = auto()
    READY_TO_CALIBRATE = auto()
    CALIBRATING = auto()
    COLLECTING = auto()


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

        print("\n--- Starting Session: Stage 2 - Data Collection ---")
        self._set_state(AppState.COLLECTING)

        self.camera_worker.camera_started.connect(self.on_camera_ready_for_recording)

        # Start all threads for the main collection loop
        self.storage_thread.start()
        self.inference_thread.started.connect(self.inference_worker.run)
        self.inference_thread.start()
        self.camera_thread.start()

    @pyqtSlot()
    def stop_session(self):
        if self.state == AppState.IDLE:
            return
        print("Controller: Stopping all active workers...")

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
        session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"session_{session_id}.mp4"
        output_path = str(config.DATA_OUTPUT_DIR / video_filename)
        self.start_recording_signal.emit(output_path, width, height, config.VIDEO_FPS)
        self.camera_worker.camera_started.disconnect(self.on_camera_ready_for_recording)

    @pyqtSlot(list)
    def on_new_result(self, results: list):
        """Handles new gaze results during the main collection phase."""
        if self.state == AppState.COLLECTING and results:
            gaze = results[0]["gaze"]
            print(
                f"Gaze Result: Pitch={gaze['pitch']:.1f}, Yaw={gaze['yaw']:.1f}",
                end="\r",
            )

    # --- Utility and Cleanup Methods ---

    def _print_session_summary(self):
        """Prints a formatted summary of the collected metadata."""
        info = self.session_metadata
        print("\n================ SESSION METADATA ================")
        print(f"OS:            {info['system_info']['os']}")
        cpu = info["system_info"]["cpu"]
        print(
            f"CPU:           {cpu['name']} ({cpu['physical_cores']}P/{cpu['total_cores']}T cores)"
        )
        ram = info["system_info"]["ram"]
        print(
            f"RAM:           {ram['total_gb']} GB Total, {ram['available_gb']} GB Available"
        )
        print(
            f"Screen Size:   {info['screen_size']['width']}x{info['screen_size']['height']}"
        )
        if "camera_resolution" in info:
            res = info["camera_resolution"]
            print(f"Camera Res:    {res['width']}x{res['height']}")
        print("--------------------------------------------------")
        print(f"INFERENCE FPS: {info['performance']['inference_fps']:.2f} FPS")
        print("==================================================")

    def on_camera_error(self, error_msg: str):
        print(f"FATAL CAMERA ERROR: {error_msg}")
        self.stop_session()

    def cleanup(self):
        print("Performing cleanup...")
        self.stop_session()
