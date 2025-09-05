"""
2D Gaze Estimation Demo with Calibration

Calibration mode: User stares at 9 dots to train linear regression mappers
Inference mode: Real-time Point of Gaze (POG) estimation
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import time
import warnings
from typing import List, Tuple

import cv2
import numpy as np

from src.inference import GazePipeline2D, GazePipeline3D, Mapper

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)


class CalibrationDemo:
    """Handles calibration UI and demo logic."""

    def __init__(self, pipeline_3d: GazePipeline3D):
        self.pipeline_3d = pipeline_3d
        self.mapper = Mapper()
        self.pipeline_2d = None  # Will be created after calibration

        # Calibration state
        self.calibration_points = self._generate_calibration_grid()
        self.current_point_idx = 0
        self.calibrating = False
        self.point_start_time = 0
        self.point_duration = 2  # seconds

        # Current point data collection
        self.current_point_gaze_vectors = []

        self.waiting_for_next = False  # for pausing during calibration

        print("Calibration demo initialized")

    def _generate_calibration_grid(
        self, w=1280, h=720, margin_ratio=0.05
    ) -> List[Tuple[int, int]]:
        """
        Generate 3x3 grid with points spread out and margin from edges.
        """
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        grid_x = np.linspace(margin_x, w - margin_x, 3)
        grid_y = np.linspace(margin_y, h - margin_y, 3)
        points = [(int(x), int(y)) for y in grid_y for x in grid_x]
        return points

    def start_calibration(self):
        """Start the calibration process."""
        self.mapper.reset()
        self.calibrating = True
        self.current_point_idx = 0
        self.point_start_time = time.time()
        self.current_point_gaze_vectors = []
        print("Starting calibration...")

    def calibration_step(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame during calibration."""
        h, w = frame.shape[:2]

        if not self.calibrating:
            return frame

        # If waiting for user to continue
        if self.waiting_for_next:
            text = "Rest & blink. Press SPACE to continue."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 3

            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Center the text
            text_x = (w - text_width) // 2
            text_y = (h + text_height) // 2

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
            )
            return frame

        if self.current_point_idx >= len(self.calibration_points):
            # Calibration complete - train mapper
            self._train_mapper()
            self.calibrating = False
            print("Calibration complete!")
            return frame

        # Current calibration point
        self.calibration_points = self._generate_calibration_grid(w, h)
        point_x, point_y = self.calibration_points[self.current_point_idx]

        current_time = time.time()
        elapsed = current_time - self.point_start_time

        # Draw calibration point
        color = (0, 0, 255) if elapsed < self.point_duration else (0, 255, 0)
        cv2.circle(frame, (point_x, point_y), 10, color, -1)
        cv2.circle(frame, (point_x, point_y), 14, (255, 255, 255), 2)

        # Collect gaze data during point display
        if elapsed < self.point_duration:
            results = self.pipeline_3d(frame)
            if results:
                # Use first detected face
                gaze = results[0]["gaze"]
                self.current_point_gaze_vectors.append([gaze["pitch"], gaze["yaw"]])
        else:
            # Point duration complete - save data and wait for user
            if self.current_point_gaze_vectors:
                # Select middle 11 frames
                n = len(self.current_point_gaze_vectors)
                if n > 11:
                    start = (n - 11) // 2
                    selected_vectors = self.current_point_gaze_vectors[
                        start : start + 11
                    ]
                else:
                    selected_vectors = self.current_point_gaze_vectors

                self.mapper.add_calibration_point(selected_vectors, (point_x, point_y))
                print(
                    f"Point {self.current_point_idx + 1}: collected {len(selected_vectors)} samples"
                )

            self.waiting_for_next = True

        # Draw progress
        progress = f"Point {self.current_point_idx + 1}/{len(self.calibration_points)}"
        cv2.putText(
            frame,
            progress,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame

    def next_calibration_point(self):
        """Advance to next calibration point after user presses key."""
        self.current_point_idx += 1
        self.point_start_time = time.time()
        self.current_point_gaze_vectors = []
        self.waiting_for_next = False

    def _train_mapper(self):
        """Train the mapper with collected calibration data."""
        try:
            score_x, score_y = self.mapper.train()
            stats = self.mapper.get_training_stats()

            print(f"Mapper trained with {stats['num_samples']} samples")
            print(f"X mapper R² score: {score_x:.3f}")
            print(f"Y mapper R² score: {score_y:.3f}")

            # Create 2D pipeline
            self.pipeline_2d = GazePipeline2D(self.pipeline_3d, self.mapper)

        except Exception as e:
            print(f"Training failed: {e}")

    def inference_step(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Perform POG inference.

        Returns:
            Tuple[np.ndarray, bool]: (processed_frame, pog_detected)
        """
        if self.pipeline_2d is None:
            return frame, False

        pog = self.pipeline_2d.predict(frame)

        if pog is not None:
            h, w = frame.shape[:2]
            # Clip coordinates to frame size
            screen_x = max(0, min(w - 1, int(pog[0])))
            screen_y = max(0, min(h - 1, int(pog[1])))

            # Draw POG
            cv2.circle(frame, (screen_x, screen_y), 7, (0, 0, 255), -1)
            cv2.circle(frame, (screen_x, screen_y), 10, (255, 255, 255), 2)
            return frame, True

        return frame, False

    def is_calibrated(self) -> bool:
        """Check if calibration is complete."""
        return self.mapper.is_trained

    def reset_tracking(self):
        """Reset tracking state."""
        self.pipeline_3d.reset_tracking()


def run_demo(
    weights_path: str,
    source: str = "0",
    device: str = "auto",
    smooth_gaze: bool = False,
):
    """Run the 2D gaze estimation demo with calibration."""
    print("Initializing 3D gaze estimation pipeline...")
    pipeline_3d = GazePipeline3D(weights_path, device=device, smooth_gaze=smooth_gaze)

    demo = CalibrationDemo(pipeline_3d)

    # Setup video capture
    try:
        source_int = int(source)
    except ValueError:
        source_int = source

    cap = cv2.VideoCapture(source_int)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # Setup for webcam
    if isinstance(source_int, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Demo controls:")
    print("'c' - Start calibration")
    print("'i' - Switch to inference mode (after calibration)")
    print("'r' - Reset and recalibrate")
    print("'q' - Quit")

    mode = "calibration"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if isinstance(source_int, int):
                frame = cv2.flip(frame, 1)

            if mode == "calibration":
                frame = demo.calibration_step(frame)

                # Draw instructions
                cv2.putText(
                    frame,
                    "CALIBRATION MODE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'c' to start calibration",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                if demo.is_calibrated():
                    cv2.putText(
                        frame,
                        "Calibrated! Press 'i' for inference",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            elif mode == "inference":
                frame, pog_detected = demo.inference_step(frame)

                cv2.putText(
                    frame,
                    "INFERENCE MODE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'r' to recalibrate",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if not pog_detected:
                    cv2.putText(
                        frame,
                        "No gaze detected",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("2D Gaze Estimation Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and mode == "calibration":
                demo.start_calibration()
            elif key == ord("i") and demo.is_calibrated():
                mode = "inference"
            elif key == ord("r"):
                mode = "calibration"
                demo.start_calibration()
            elif key == ord("t"):  # 't' for tracking reset
                demo.reset_tracking()
            elif key == ord(" "):  # SPACE pressed
                if mode == "calibration" and demo.waiting_for_next:
                    demo.next_calibration_point()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="2D Gaze Estimation Demo with Calibration"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to 3D model weights"
    )
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--smooth-gaze", action="store_true", help="Enable gaze smoothing"
    )

    args = parser.parse_args()
    run_demo(args.weights, args.source, args.device, args.smooth_gaze)


if __name__ == "__main__":
    main()
