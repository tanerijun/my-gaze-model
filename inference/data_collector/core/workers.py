import time
from queue import Empty, Queue

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.inference import GazePipeline3D


class CameraWorker(QObject):
    """
    Captures video frames from the camera in a separate thread.
    Put frames into queues for consumption.
    """

    frame_ready = pyqtSignal(np.ndarray)
    camera_started = pyqtSignal(int, int)  # emits width and height
    camera_error = pyqtSignal(str)

    def __init__(
        self,
        camera_id: int,
        desired_resolution: tuple,
        video_queue: Queue,
        inference_queue: Queue,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.desired_resolution = desired_resolution
        self.video_queue = video_queue
        self.inference_queue = inference_queue
        self._is_running = False
        self._enable_video_recording = True

    def set_video_recording(self, enabled: bool):
        """Enable or disable writing frames to the video queue."""
        self._enable_video_recording = enabled

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

            # Emit the raw frame for any listeners (like the benchmark)
            self.frame_ready.emit(frame)

            if self._enable_video_recording:
                self.video_queue.put(frame)

            self.inference_queue.put(frame)

        cap.release()
        print("Camera worker stopped.")

    def stop(self):
        """Stops the capture loop."""
        self._is_running = False


class InferenceWorker(QObject):
    """
    Runs the GazePipeline3D model in a separate thread.
    Pulls frames from a queue and runs the gaze model.
    """

    result_ready = pyqtSignal(list)
    benchmark_finished = pyqtSignal(float)  # dedicated benchmark is done

    def __init__(self, weights_path: str, config: dict, inference_queue: Queue):
        super().__init__()
        self.weights_path = weights_path
        self.config = config
        self.inference_queue = inference_queue
        self.pipeline = None
        self._is_running = False

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

    @pyqtSlot()
    def run(self):
        """Starts the inference loop."""
        self._is_running = True
        print("Inference loop started.")

        while self._is_running:
            try:
                # Get the latest frame from the queue, discarding older ones.
                # This ensures the lowest possible latency.
                frame = None
                while not self.inference_queue.empty():
                    frame = self.inference_queue.get_nowait()

                if frame is not None and self.pipeline is not None:
                    results = self.pipeline(frame)
                    if results:
                        self.result_ready.emit(results)
                else:
                    # If queue is empty, wait a moment to avoid busy-waiting
                    time.sleep(0.01)
            except Empty:
                time.sleep(0.01)  # Wait if the queue is empty
            except Exception as e:
                print(f"Error during inference: {e}")

        print("Inference loop stopped.")

    @pyqtSlot(np.ndarray, int)
    def run_benchmark(self, frame: np.ndarray, num_frames: int):
        """Runs a blocking benchmark on a single frame."""
        if not self.pipeline:
            print("Cannot run benchmark, pipeline not initialized.")
            self.benchmark_finished.emit(-1.0)
            return

        print(f"Starting dedicated benchmark for {num_frames} iterations...")
        start_time = time.monotonic()

        for i in range(num_frames):
            _ = self.pipeline(frame)
            print(f"Benchmarking... Frame {i + 1}/{num_frames}", end="\r")

        duration = time.monotonic() - start_time
        fps = num_frames / duration
        print(f"\nDedicated benchmark finished. FPS: {fps:.2f}")
        self.benchmark_finished.emit(fps)

    def stop(self):
        self._is_running = False


class StorageWorker(QObject):
    """Pulls frames from a queue and writes them to a video file."""

    recording_finished = pyqtSignal(str)
    storage_error = pyqtSignal(str)

    def __init__(self, video_queue: Queue):
        super().__init__()
        self.video_queue = video_queue
        self._is_running = False
        self.video_writer = None
        self.output_path = ""

    @pyqtSlot(str, int, int, int)
    def setup_and_run(self, output_path: str, width: int, height: int, fps: int):
        """Sets up the VideoWriter AND starts the blocking run loop."""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.video_writer.isOpened():
            print("AVC1 codec failed, trying MP4V...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )

        if not self.video_writer.isOpened():
            self.storage_error.emit(
                f"Could not open video writer for path: {output_path}"
            )
            return

        print(f"Storage worker started recording to: {self.output_path}")

        self._is_running = True
        while self._is_running or not self.video_queue.empty():
            try:
                frame = self.video_queue.get(timeout=0.1)
                self.video_writer.write(frame)
            except Empty:
                # If the queue is empty and we've been told to stop, exit the loop
                if not self._is_running:
                    break
                continue
            except Exception as e:
                self.storage_error.emit(f"Error in storage worker: {e}")
                self._is_running = False  # Stop on error

        if self.video_writer:
            self.video_writer.release()
        self.video_writer = None
        print(f"Video saved: {self.output_path}")
        self.recording_finished.emit(self.output_path)
        print("Storage worker loop stopped.")

    def stop(self):
        """Signals the worker to finish up and stop."""
        print("Storage worker received stop signal.")
        self._is_running = False
