import datetime
from queue import Queue

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from data_collector import config
from data_collector.core.workers import CameraWorker, InferenceWorker, StorageWorker


class AppController(QObject):
    """
    Orchestrates all workers and threads for the application.
    """

    # Signal to command the storage worker to start
    start_recording_signal = pyqtSignal(str, int, int, int)

    def __init__(self):
        super().__init__()
        self.is_collecting = False
        self.session_metadata = {}

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

    @pyqtSlot()
    def start_collection(self):
        if self.is_collecting:
            return

        config.DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Controller: Starting collection...")
        self.is_collecting = True
        self.session_metadata = {}

        self.camera_thread.start()
        self.inference_thread.start()
        self.storage_thread.start()

    @pyqtSlot()
    def stop_collection(self):
        if not self.is_collecting:
            return
        print("Controller: Stopping collection...")
        self.is_collecting = False

        # Signal all workers to stop
        self.camera_worker.stop()
        self.inference_worker.stop()
        self.storage_worker.stop()

        # Quit the threads and wait for them to finish
        for thread in [self.camera_thread, self.inference_thread, self.storage_thread]:
            thread.quit()
            thread.wait()

        # Clear queues in case there's leftover data
        while not self.video_queue.empty():
            self.video_queue.get_nowait()
        while not self.inference_queue.empty():
            self.inference_queue.get_nowait()

        print("Controller: All threads have stopped.")

    @pyqtSlot(int, int)
    def on_camera_started(self, width: int, height: int):
        """Stores the actual camera resolution when the camera starts."""
        print(f"Controller: Received actual camera resolution: {width}x{height}")
        self.session_metadata["camera_resolution"] = {"width": width, "height": height}

        # start storage worker
        session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"session_{session_id}.mp4"
        output_path = str(config.DATA_OUTPUT_DIR / video_filename)

        self.start_recording_signal.emit(output_path, width, height, config.VIDEO_FPS)

    @pyqtSlot(list)
    def on_new_result(self, results: list):
        """Handles new gaze estimation results from the inference worker."""
        # For now, just print the first result to confirm data is flowing
        if results:
            pass
            # first_face = results[0]
            # gaze = first_face["gaze"]
            # head_pose = first_face["gaze_origin_features"]
            # print(
            #     f"Gaze: P={gaze['pitch']:.1f}, Y={gaze['yaw']:.1f} | "
            #     f"Head Roll: {head_pose.get('roll_angle', 0.0):.1f}"
            # )

    def on_camera_error(self, error_msg: str):
        print(f"FATAL CAMERA ERROR: {error_msg}")
        # Maybe show a message box to the user
        self.stop_collection()  # Stop everything if camera fails

    def cleanup(self):
        """Ensures threads are stopped gracefully on application exit."""
        print("Performing cleanup...")
        self.stop_collection()
