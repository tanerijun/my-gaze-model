import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.inference import GazePipeline3D


class CameraWorker(QObject):
    """Captures video frames from the camera in a separate thread."""

    frame_ready = pyqtSignal(np.ndarray)
    camera_started = pyqtSignal(int, int)  # emits width and height
    camera_error = pyqtSignal(str)

    def __init__(self, camera_id: int, desired_resolution: tuple[int, int]):
        super().__init__()
        self.camera_id = camera_id
        self.desired_resolution = desired_resolution
        self._is_running = False

    @pyqtSlot()
    def run(self):
        """Starts the camera capture loop."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.camera_error.emit(f"Could not open camera with ID {self.camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_resolution[1])

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == 0 or actual_height == 0:
            self.camera_error.emit(
                f"Camera {self.camera_id} returned invalid resolution (0x0)."
            )
            cap.release()
            return

        # Emit actual resolution back to controller
        self.camera_started.emit(actual_width, actual_height)

        self._is_running = True
        print(f"Camera worker started with resolution: {actual_width}x{actual_height}")

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)  # Avoid busy-waiting if the camera stalls
                continue

            # Flip the frame horizontally for a mirror effect, common for webcams
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame)

        cap.release()
        print("Camera worker stopped.")

    def stop(self):
        """Stops the capture loop."""
        self._is_running = False


class InferenceWorker(QObject):
    """Runs the GazePipeline3D model in a separate thread."""

    result_ready = pyqtSignal(list)

    def __init__(self, weights_path: str, config: dict):
        super().__init__()
        self.weights_path = weights_path
        self.config = config
        self.pipeline = None

    def setup(self):
        """Initializes the pipeline. Called after moving to the thread."""
        try:
            self.pipeline = GazePipeline3D(
                weights_path=str(self.weights_path),
                device="auto",  # Let the pipeline decide cuda/cpu
                enable_landmarker_features=self.config["enable_landmarker_features"],
                smooth_facebbox=self.config["smooth_facebbox"],
                smooth_gaze=self.config["smooth_gaze"],
            )
            print("Inference worker and pipeline initialized.")
        except Exception as e:
            print(f"Error initializing GazePipeline3D: {e}")
            self.pipeline = None

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame: np.ndarray):
        """Processes a single frame and emits the results."""
        if self.pipeline is None:
            return

        try:
            results = self.pipeline(frame)
            if results:  # Only emit if a face was detected
                self.result_ready.emit(results)
        except Exception as e:
            print(f"Error during inference: {e}")
