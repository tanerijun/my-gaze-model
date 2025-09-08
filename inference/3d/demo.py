"""
Streamlined Gaze Estimation Demo

A production-ready inference demo for gaze estimation using MobileOne S1.
This script provides real-time gaze estimation with webcam input.

Usage:
    python demo.py --weights path/to/model.pth
    python demo.py --weights path/to/model.pth --source 0  # default webcam
    python demo.py --weights path/to/model.pth --source video.mp4  # video file
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import queue
import threading
import time

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh_connections

from src import GazePipeline3D


def draw_bbox_and_gaze(frame: np.ndarray, result: dict):
    """Draws the gaze vector using the ORIGINAL, PROVEN mathematical formula."""
    if "bbox" not in result or "gaze" not in result:
        return

    bbox = result["bbox"]
    gaze = result["gaze"]

    # Draw bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    pitch_rad = np.radians(gaze["pitch"])
    yaw_rad = np.radians(gaze["yaw"])
    length = 150

    dx = -length * np.sin(pitch_rad) * np.cos(yaw_rad)
    dy = -length * np.sin(yaw_rad)

    end_point = (int(center_x + dx), int(center_y + dy))

    cv2.arrowedLine(
        frame, (center_x, center_y), end_point, (0, 0, 255), 2, tipLength=0.2
    )


def draw_head_pose(frame: np.ndarray, result: dict, cam_matrix: np.ndarray):
    """Draws the 3D head pose axis."""
    if (
        "head_pose_matrix" not in result
        or result["head_pose_matrix"] is None
        or "mediapipe_landmarks" not in result
        or result["mediapipe_landmarks"] is None
    ):
        # We need both the matrix for direction and landmarks for origin
        return

    h, w, _ = frame.shape
    head_pose_matrix = result["head_pose_matrix"]
    landmarks = result["mediapipe_landmarks"]

    # Project 3D axes points to 2D
    axis_points = np.array([[50, 0, 0], [0, -50, 0], [0, 0, 50]], dtype=np.float32)
    rotation_matrix = head_pose_matrix[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    translation_vector = head_pose_matrix[:3, 3]
    rotation_vector[0] *= -1

    imgpts, _ = cv2.projectPoints(
        axis_points,
        rotation_vector,
        translation_vector,
        cam_matrix,
        np.zeros((4, 1)),
    )
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Use nose tip (landmark #1) as the origin for the axes
    origin = (int(landmarks[1].x * w), int(landmarks[1].y * h))

    cv2.line(frame, origin, tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis: Red
    cv2.line(frame, origin, tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis: Green
    cv2.line(frame, origin, tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis: Blue


def draw_blaze_keypoints(frame: np.ndarray, result: dict):
    """Draws the 6 keypoints from the BlazeFace detector."""
    if "blaze_keypoints" not in result or result["blaze_keypoints"] is None:
        return

    h, w, _ = frame.shape
    for kp in result["blaze_keypoints"]:
        center = (int(kp.x * w), int(kp.y * h))
        cv2.circle(frame, center, 2, (255, 255, 0), -1)  # Cyan dots


def draw_mediapipe_mesh(frame: np.ndarray, result: dict):
    """Draws the full 478-point face mesh from MediaPipe landmarks."""
    if "mediapipe_landmarks" not in result or result["mediapipe_landmarks"] is None:
        return

    landmarks = result["mediapipe_landmarks"]
    h, w, _ = frame.shape

    # Define the drawing spec for the mesh connections
    connection_spec = {"color": (224, 224, 224), "thickness": 1}

    # FACEMESH_TESSELATION is a set of (start_index, end_index) tuples
    for connection in face_mesh_connections.FACEMESH_TESSELATION:
        start_idx = connection[0]
        end_idx = connection[1]

        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        # Denormalize coordinates to get pixel values
        start_pixel = (int(start_point.x * w), int(start_point.y * h))
        end_pixel = (int(end_point.x * w), int(end_point.y * h))

        # Draw the line on the frame
        cv2.line(
            frame,
            start_pixel,
            end_pixel,
            connection_spec["color"],
            connection_spec["thickness"],
        )

    # OPTIONAL: Draw the individual landmark points
    point_spec = {"color": (0, 255, 0), "thickness": -1, "radius": 1}
    for landmark in landmarks:
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        cv2.circle(
            frame,
            (x_px, y_px),
            point_spec["radius"],
            point_spec["color"],
            point_spec["thickness"],
        )


def draw_gaze_origin(frame: np.ndarray, result: dict, text_buffer: list):
    """Draws graphical elements and adds text info to a buffer for deferred rendering."""
    if "gaze_origin_features" not in result or "blaze_keypoints" not in result:
        return

    h, w, _ = frame.shape
    features, keypoints = result["gaze_origin_features"], result["blaze_keypoints"]

    # Draw graphical elements that need to be flipped
    right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))
    cv2.circle(frame, right_eye, 3, (255, 0, 255), -1)
    cv2.circle(frame, left_eye, 3, (255, 0, 255), -1)
    cv2.line(frame, right_eye, left_eye, (255, 0, 255), 1)

    # Prepare text and its position on the UN-FLIPPED frame
    text = f"IPD: {features['ipd']:.2f}, Roll: {features['roll_angle']:.1f}"
    pos = (result["bbox"][0], result["bbox"][1] - 10)

    # Add all info needed for rendering to the buffer
    text_buffer.append(
        {
            "text": text,
            "pos": pos,
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "scale": 0.5,
            "color": (255, 255, 255),
            "thickness": 1,
        }
    )


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


def run_demo(args):
    """
    Run the gaze estimation demo.
    """
    use_landmarker = args.show_head_pose or args.show_mediapipe_mesh
    if use_landmarker:
        print("Enabling MediaPipe FaceLandmarker for requested visualizations.")

    pipeline = GazePipeline3D(
        args.weights,
        device=args.device,
        smooth_facebbox=args.smooth_facebbox,
        smooth_gaze=args.smooth_gaze,
        enable_landmarker_features=use_landmarker,
    )

    frame_producer = FrameProducer(args.source)
    is_webcam = args.source.isdigit()
    # Camera intrinsics
    cam_matrix = None

    try:
        frame_producer.start()
        print(f"Started video capture from: {args.source}")
        print("Press 'q' to quit, 'r' to reset tracking")

        while True:
            start_time = time.time()

            frame = frame_producer.get_frame()
            if frame is None:
                continue

            # Estimate camera matrix on first frame
            if cam_matrix is None:
                h, w = frame.shape[:2]
                # Simple approximation of the camera intrinsic matrix
                cam_matrix = np.array(
                    [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32
                )

            results = pipeline(frame)
            text_buffer = []
            for result in results:
                draw_bbox_and_gaze(frame, result)

                if args.show_head_pose:
                    draw_head_pose(frame, result, cam_matrix)

                if args.show_blaze_keypoints:
                    draw_blaze_keypoints(frame, result)

                if args.show_mediapipe_mesh:
                    draw_mediapipe_mesh(frame, result)

                if args.show_gaze_origin:
                    draw_gaze_origin(frame, result, text_buffer)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            text_buffer.append(
                {
                    "text": f"FPS: {fps:.1f}",
                    "pos": (10, 30),
                    "font": cv2.FONT_HERSHEY_SIMPLEX,
                    "scale": 1,
                    "color": (0, 255, 0),
                    "thickness": 2,
                }
            )

            if is_webcam:
                frame = cv2.flip(frame, 1)

            for text_info in text_buffer:
                cv2.putText(
                    frame,
                    text_info["text"],
                    text_info["pos"],
                    text_info["font"],
                    text_info["scale"],
                    text_info["color"],
                    text_info["thickness"],
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
    parser = argparse.ArgumentParser(description="Real-time 3D gaze estimation demo.")

    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (webcam index or file path).",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument(
        "--smooth-facebbox",
        action="store_true",
        help="Enable Kalman filtering for face bbox.",
    )
    parser.add_argument(
        "--smooth-gaze",
        action="store_true",
        help="Enable Kalman filtering for gaze vectors.",
    )

    parser.add_argument(
        "--show-head-pose", action="store_true", help="Visualize the 3D head pose axis."
    )
    parser.add_argument(
        "--show-blaze-keypoints",
        action="store_true",
        help="Visualize the 6 BlazeFace keypoints.",
    )
    parser.add_argument(
        "--show-mediapipe-mesh",
        action="store_true",
        help="Visualize the full MediaPipe face mesh.",
    )
    parser.add_argument(
        "--show-gaze-origin",
        action="store_true",
        help="Visualize the gaze origin features (eye centers, Inter-Pupillary Distance, Roll).",
    )

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
