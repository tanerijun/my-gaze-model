"""
Test client for the Gaze Estimation REST API

This script demonstrates how to use the REST API endpoints for gaze estimation.
It provides examples for both file upload and base64 image submission.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import requests


class GazeAPIClient:
    """Client for interacting with the Gaze Estimation REST API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the gaze estimation service
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Health check failed: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Status request failed: {e}")
            return {}

    def reset_tracking(self) -> Dict[str, Any]:
        """Reset tracking state."""
        try:
            response = self.session.post(f"{self.base_url}/reset")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Reset tracking failed: {e}")
            return {}

    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        """
        Predict gaze from image file.

        Args:
            image_path: Path to image file

        Returns:
            API response containing gaze predictions
        """
        try:
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                response = self.session.post(f"{self.base_url}/predict", files=files)
                response.raise_for_status()
                return response.json()
        except (requests.RequestException, FileNotFoundError) as e:
            print(f"File prediction failed: {e}")
            return {}

    def predict_from_base64(self, image_path: str) -> Dict[str, Any]:
        """
        Predict gaze from base64 encoded image.

        Args:
            image_path: Path to image file

        Returns:
            API response containing gaze predictions
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Send request
            payload = {"image": image_b64}
            response = self.session.post(
                f"{self.base_url}/predict_base64",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, FileNotFoundError) as e:
            print(f"Base64 prediction failed: {e}")
            return {}

    def predict_from_camera(self, camera_index: int = 0, duration: float = 10.0):
        """
        Predict gaze from camera feed for specified duration.

        Args:
            camera_index: Camera index (0 for default camera)
            duration: Duration in seconds to capture frames
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            return

        print(f"Starting camera test for {duration} seconds...")
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)

                # Encode frame as JPEG
                _, buffer = cv2.imencode(".jpg", frame)
                image_b64 = base64.b64encode(buffer).decode("utf-8")

                # Send to API
                payload = {"image": image_b64}
                try:
                    response = self.session.post(
                        f"{self.base_url}/predict_base64",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        result = response.json()

                        # Draw results on frame
                        for detection in result["results"]:
                            bbox = detection["bbox"]
                            gaze = detection["gaze"]

                            # Draw bounding box
                            cv2.rectangle(
                                frame,
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (0, 255, 0),
                                2,
                            )

                            # Draw gaze vector
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)

                            # Convert angles to radians for visualization
                            import math

                            pitch_rad = math.radians(gaze["pitch"])
                            yaw_rad = math.radians(gaze["yaw"])

                            # Calculate gaze vector endpoint
                            length = 100
                            dx = -length * math.sin(pitch_rad) * math.cos(yaw_rad)
                            dy = -length * math.sin(yaw_rad)

                            end_point = (int(center_x + dx), int(center_y + dy))
                            cv2.arrowedLine(
                                frame, (center_x, center_y), end_point, (0, 0, 255), 2
                            )

                            # Draw text
                            text = f"P:{gaze['pitch']:.1f}, Y:{gaze['yaw']:.1f}"
                            cv2.putText(
                                frame,
                                text,
                                (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )

                        # Show processing time
                        proc_time = result.get("processing_time", 0)
                        cv2.putText(
                            frame,
                            f"Process: {proc_time:.1f}ms",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                except requests.RequestException as e:
                    print(f"API request failed: {e}")

                # Display frame
                cv2.imshow("Gaze API Test", frame)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def benchmark_api(self, image_path: str, num_requests: int = 10):
        """
        Benchmark API performance.

        Args:
            image_path: Path to test image
            num_requests: Number of requests to send
        """
        print(f"Benchmarking API with {num_requests} requests...")

        # Read and encode image once
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {"image": image_b64}

        times = []
        successful_requests = 0

        for i in range(num_requests):
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/predict_base64",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    successful_requests += 1
                    request_time = time.time() - start_time
                    times.append(request_time)

                    result = response.json()
                    processing_time = result.get("processing_time", 0)
                    print(
                        f"Request {i + 1}: {request_time * 1000:.1f}ms total, {processing_time:.1f}ms processing"
                    )
                else:
                    print(f"Request {i + 1} failed with status {response.status_code}")

            except requests.RequestException as e:
                print(f"Request {i + 1} failed: {e}")

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print("\nBenchmark Results:")
            print(f"Successful requests: {successful_requests}/{num_requests}")
            print(f"Average response time: {avg_time * 1000:.1f}ms")
            print(f"Min response time: {min_time * 1000:.1f}ms")
            print(f"Max response time: {max_time * 1000:.1f}ms")
            print(f"Average throughput: {1 / avg_time:.1f} requests/second")


def main():
    parser = argparse.ArgumentParser(
        description="Test client for Gaze Estimation REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_rest_client.py --health                          # Check service health
  python test_rest_client.py --status                          # Get service status
  python test_rest_client.py --image photo.jpg                 # Test with image file
  python test_rest_client.py --image photo.jpg --base64        # Test with base64 encoding
  python test_rest_client.py --camera                          # Test with camera feed
  python test_rest_client.py --benchmark photo.jpg             # Benchmark API performance
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the gaze estimation service (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--health",
        action="store_true",
        help="Check service health",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Get service status",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for prediction",
    )

    parser.add_argument(
        "--base64",
        action="store_true",
        help="Use base64 encoding (default: file upload)",
    )

    parser.add_argument(
        "--camera",
        action="store_true",
        help="Test with camera feed",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for camera test (default: 0)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds for camera test (default: 10.0)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to image file for benchmarking",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests for benchmarking (default: 10)",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset tracking state",
    )

    args = parser.parse_args()

    # Create client
    client = GazeAPIClient(args.url)

    # Execute requested actions
    if args.health:
        print("Checking service health...")
        result = client.health_check()
        print(json.dumps(result, indent=2))

    elif args.status:
        print("Getting service status...")
        result = client.get_status()
        print(json.dumps(result, indent=2))

    elif args.reset:
        print("Resetting tracking state...")
        result = client.reset_tracking()
        print(json.dumps(result, indent=2))

    elif args.image:
        print(f"Processing image: {args.image}")
        if args.base64:
            print("Using base64 encoding...")
            result = client.predict_from_base64(args.image)
        else:
            print("Using file upload...")
            result = client.predict_from_file(args.image)
        print(json.dumps(result, indent=2))

    elif args.camera:
        print(
            f"Starting camera test (camera {args.camera_index}, duration {args.duration}s)"
        )
        print("Press 'q' to quit early")
        client.predict_from_camera(args.camera_index, args.duration)

    elif args.benchmark:
        client.benchmark_api(args.benchmark, args.num_requests)

    else:
        print("No action specified. Use --help for available options.")


if __name__ == "__main__":
    main()
