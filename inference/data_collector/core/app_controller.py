from PyQt6.QtCore import QObject, QThread, pyqtSlot

from data_collector import config
from data_collector.core.workers import CameraWorker, InferenceWorker


class AppController(QObject):
    """
    Orchestrates the camera and inference workers in separate threads.
    """

    def __init__(self):
        super().__init__()
        self.is_collecting = False
        self.session_metadata = {}

        # --- Setup Camera Thread ---
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(
            config.CAMERA_ID, config.DESIRED_CAMERA_RESOLUTION
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.camera_error.connect(self.on_camera_error)
        self.camera_worker.camera_started.connect(self.on_camera_started)

        # --- Setup Inference Thread ---
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker(
            str(config.GAZE_MODEL_PATH), config.PIPELINE_CONFIG
        )
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.started.connect(self.inference_worker.setup)
        self.inference_worker.result_ready.connect(self.on_new_result)

        # --- Connect Workers ---
        # Connect the frame signal from camera to the processing slot in inference
        self.camera_worker.frame_ready.connect(self.inference_worker.process_frame)

        print("AppController initialized.")

    @pyqtSlot()
    def start_collection(self):
        if self.is_collecting:
            return
        print("Controller: Starting worker threads...")
        self.is_collecting = True
        self.session_metadata = {}
        self.camera_thread.start()
        self.inference_thread.start()

    @pyqtSlot()
    def stop_collection(self):
        if not self.is_collecting:
            return
        print("Controller: Stopping worker threads...")
        self.is_collecting = False
        self.camera_worker.stop()
        # Quit the threads and wait for them to finish
        self.camera_thread.quit()
        self.camera_thread.wait()
        self.inference_thread.quit()
        self.inference_thread.wait()
        print("Controller: Worker threads have stopped.")

    @pyqtSlot(int, int)
    def on_camera_started(self, width: int, height: int):
        """Stores the actual camera resolution when the camera starts."""
        print(f"Controller: Received actual camera resolution: {width}x{height}")
        self.session_metadata["camera_resolution"] = {"width": width, "height": height}

    @pyqtSlot(list)
    def on_new_result(self, results: list):
        """Handles new gaze estimation results from the inference worker."""
        # For now, just print the first result to confirm data is flowing
        if results:
            first_face = results[0]
            gaze = first_face["gaze"]
            head_pose = first_face["gaze_origin_features"]
            print(
                f"Gaze: P={gaze['pitch']:.1f}, Y={gaze['yaw']:.1f} | "
                f"Head Roll: {head_pose.get('roll_angle', 0.0):.1f}"
            )

    def on_camera_error(self, error_msg: str):
        print(f"FATAL CAMERA ERROR: {error_msg}")
        # Maybe show a message box to the user
        self.stop_collection()  # Stop everything if camera fails

    def cleanup(self):
        """Ensures threads are stopped gracefully on application exit."""
        print("Performing cleanup...")
        self.stop_collection()
