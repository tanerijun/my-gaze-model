#!/usr/bin/env python3
"""
Streamlined Gaze Estimation Demo

A production-ready inference demo for gaze estimation using MobileOne S1.
This script provides real-time gaze estimation with webcam input.

Usage:
    python demo.py --weights path/to/model.pth
    python demo.py --weights path/to/model.pth --source 0  # default webcam
    python demo.py --weights path/to/model.pth --source video.mp4  # video file
"""

import argparse
import queue
import threading
import time
from typing import Tuple

import cv2
import numpy as np

from src import GazePipeline


def draw_gaze(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pitch: float,
    yaw: float,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw gaze vector on the image.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        color: Arrow color in BGR format

    Returns:
        Image with gaze vector drawn
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Calculate gaze vector endpoint
    length = 150  # Length of the gaze vector
    dx = -length * np.sin(pitch_rad) * np.cos(yaw_rad)
    dy = -length * np.sin(yaw_rad)

    end_point = (int(center_x + dx), int(center_y + dy))

    # Draw the gaze vector
    cv2.arrowedLine(image, (center_x, center_y), end_point, color, 2, tipLength=0.2)

    # Draw angle text
    text = f"P:{pitch:.1f}, Y:{yaw:.1f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


class FrameProducer:
    """Threaded frame producer for smooth video capture."""

    def __init__(self, video_source, max_queue_size: int = 5):
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.cap = None
        self.thread = None

    def start(self):
        """Start the frame producer thread."""
        try:
            # Try to convert to int for webcam
            source = int(self.video_source)
        except ValueError:
            # Use as string for video file
            source = self.video_source

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_source}")

        self.thread = threading.Thread(target=self._produce_frames)
        self.thread.start()

    def _produce_frames(self):
        """Frame production loop."""
        while not self.stop_event.is_set() and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break

            # Flip frame for webcam (mirror effect)
            if isinstance(self.video_source, int) or self.video_source.isdigit():
                frame = cv2.flip(frame, 1)

            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                # Drop frame if queue is full
                continue

    def get_frame(self, timeout: float = 0.5):
        """Get the next frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the frame producer."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


def run_demo(
    weights_path: str,
    source: str = "0",
    device: str = "auto",
    smooth_gaze: bool = False,
):
    """
    Run the gaze estimation demo.

    Args:
        weights_path: Path to the trained model weights
        source: Video source (webcam index or video file path)
        device: Compute device ("cpu", "cuda", or "auto")
        smooth_gaze: Enable Kalman filtering for gaze vectors
    """
    print("Initializing gaze estimation pipeline...")
    pipeline = GazePipeline(weights_path, device=device, smooth_gaze=smooth_gaze)

    print("Setting up video capture...")
    frame_producer = FrameProducer(source)

    try:
        frame_producer.start()
        print(f"Started video capture from: {source}")
        print("Press 'q' to quit, 'r' to reset tracking")

        while True:
            start_time = time.time()

            # Get next frame
            frame = frame_producer.get_frame()
            if frame is None:
                continue

            # Process frame for gaze estimation
            results = pipeline(frame)

            # Draw results
            for result in results:
                bbox = result["bbox"]
                gaze = result["gaze"]

                # Draw bounding box
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                )

                # Draw gaze vector
                frame = draw_gaze(frame, bbox, gaze["pitch"], gaze["yaw"])

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display frame
            cv2.imshow("Gaze Estimation Demo", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                pipeline.reset_tracking()
                print("Tracking reset")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        print("Cleaning up...")
        frame_producer.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time gaze estimation demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --weights model.pth                     # Use default webcam
  python demo.py --weights model.pth --source 1          # Use webcam index 1
  python demo.py --weights model.pth --source video.mp4  # Use video file
  python demo.py --weights model.pth --device cpu        # Force CPU inference
  python demo.py --weights model.pth --smooth-gaze       # Enable gaze smoothing
        """,
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth file)",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0, 1, ...) or path to video file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for inference",
    )

    parser.add_argument(
        "--smooth-gaze",
        action="store_true",
        help="Enable Kalman filtering for gaze vectors (smoother but less responsive)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Gaze Estimation Demo")
    print("=" * 50)
    print(f"Model weights: {args.weights}")
    print(f"Video source: {args.source}")
    print(f"Device: {args.device}")
    print(f"Gaze smoothing: {args.smooth_gaze}")
    print("=" * 50)

    run_demo(args.weights, args.source, args.device, args.smooth_gaze)


if __name__ == "__main__":
    main()
