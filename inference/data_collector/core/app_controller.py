import datetime
import time
from enum import Enum, auto
from queue import Queue

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
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
        self.camera_worker.camera_started.connect(self.on_camera_started)

        # --- Setup Inference Worker & Thread ---
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker(
            str(config.GAZE_MODEL_PATH), config.PIPELINE_CONFIG, self.inference_queue
        )
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.started.connect(self.inference_worker.setup)
        self.inference_thread.started.connect(self.inference_worker.run)
        self.inference_worker.result_ready.connect(self.on_new_result)

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

        self.benchmark_frame_count = 0
        self.benchmark_start_time = 0.0

        self.session_metadata["system_info"] = get_system_info()

        screen = QApplication.primaryScreen()
        if not screen:
            raise ValueError("screen is None")

        self.session_metadata["screen_size"] = {
            "width": screen.size().width(),
            "height": screen.size().height(),
        }

        self.camera_thread.start()
        self.inference_thread.start()

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
            if thread.isRunning:
                thread.quit()
                thread.wait()

        # Clear queues in case there's leftover data
        while not self.video_queue.empty():
            self.video_queue.get_nowait()
        while not self.inference_queue.empty():
            self.inference_queue.get_nowait()

        self._set_state(AppState.IDLE)
        print("Controller: All processes stopped. Ready for new session.")

    @pyqtSlot(list)
    def on_new_result(self, results: list):
        """Handles results from the inference worker based on the current state."""
        if self.state == AppState.BENCHMARKING:
            self._handle_benchmark_result()
        elif self.state == AppState.COLLECTING:
            if results:
                gaze = results[0]["gaze"]
                print(
                    f"Gaze Result: Pitch={gaze['pitch']:.1f}, Yaw={gaze['yaw']:.1f}",
                    end="\r",
                )

    def _handle_benchmark_result(self):
        """Processes one frame for the benchmark calculation."""
        if self.benchmark_frame_count == 0:
            # Start timer on the first processed frame
            self.benchmark_start_time = time.monotonic()

        self.benchmark_frame_count += 1
        print(
            f"Benchmarking... Frame {self.benchmark_frame_count}/{config.BENCHMARK_FRAMES}",
            end="\r",
        )

        if self.benchmark_frame_count >= config.BENCHMARK_FRAMES:
            duration = time.monotonic() - self.benchmark_start_time
            fps = self.benchmark_frame_count / duration
            self.session_metadata["performance"] = {"pipeline_fps": round(fps, 2)}

            print("\n--- Benchmarking Complete ---")
            self._print_session_summary()

            # Stop the workers used for benchmarking
            self.camera_worker.stop()
            self.inference_worker.stop()
            self.camera_thread.quit()
            self.camera_thread.wait()
            self.inference_thread.quit()
            self.inference_thread.wait()

            self._set_state(AppState.READY_TO_CALIBRATE)
            print("\nReady to start calibration.")

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
        print(f"PIPELINE FPS:  {info['performance']['pipeline_fps']:.2f} FPS")
        print("==================================================")

    @pyqtSlot(int, int)
    def on_camera_started(self, width: int, height: int):
        """Stores camera resolution. During collection, it also starts recording."""
        print(f"Controller: Received actual camera resolution: {width}x{height}")
        self.session_metadata["camera_resolution"] = {"width": width, "height": height}

        if self.state == AppState.COLLECTING:
            # This logic will be used when we start the actual data collection
            session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = f"session_{session_id}.mp4"
            output_path = str(config.DATA_OUTPUT_DIR / video_filename)
            self.start_recording_signal.emit(
                output_path, width, height, config.VIDEO_FPS
            )

    def on_camera_error(self, error_msg: str):
        print(f"FATAL CAMERA ERROR: {error_msg}")
        self.stop_session()

    def cleanup(self):
        print("Performing cleanup...")
        self.stop_session()
