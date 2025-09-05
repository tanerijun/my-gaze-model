"""
Example Integration Script for Gaze Estimation API

This script demonstrates how to integrate the gaze estimation service
into a real-world application with proper error handling, logging,
and performance monitoring.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import base64
import logging
import time
from typing import Dict, List, Optional

import aiohttp
import cv2
import numpy as np


class GazeEstimationClient:
    """
    Async client for gaze estimation service with connection pooling
    and automatic retry logic.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize the gaze estimation client.

        Args:
            base_url: Base URL of the gaze estimation service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None

        # Performance tracking
        self.total_requests = 0
        self.total_errors = 0
        self.total_processing_time = 0.0

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def health_check(self) -> bool:
        """
        Check if the service is healthy and responsive.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            if self.session:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "healthy"
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
        return False

    async def predict_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Predict gaze from video frame with automatic retry.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            Prediction results or None if failed
        """
        # Encode frame as JPEG
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            self.logger.error("Failed to encode frame")
            return None

        # Convert to base64
        image_b64 = base64.b64encode(buffer).decode("utf-8")
        payload = {"image": image_b64}

        # Attempt prediction with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                if self.session:
                    async with self.session.post(
                        f"{self.base_url}/predict_base64",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        request_time = time.time() - start_time
                        self.total_requests += 1

                        if response.status == 200:
                            result = await response.json()
                            self.total_processing_time += request_time
                            return result
                        else:
                            self.logger.warning(
                                f"API returned status {response.status} on attempt {attempt + 1}"
                            )

            except asyncio.TimeoutError:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
            except Exception as e:
                self.logger.error(f"Request failed on attempt {attempt + 1}: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.1 * (2**attempt))

        self.total_errors += 1
        return None

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.total_requests == 0:
            return {"error_rate": 0.0, "avg_response_time": 0.0}

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_requests * 100,
            "avg_response_time": self.total_processing_time
            / self.total_requests
            * 1000,  # ms
        }


class GazeVisualizer:
    """Helper class for visualizing gaze results on video frames."""

    def __init__(self):
        self.colors = {
            "bbox": (0, 255, 0),  # Green
            "gaze": (0, 0, 255),  # Red
            "text": (255, 255, 255),  # White
        }

    def draw_results(
        self, frame: np.ndarray, results: List[Dict], processing_time: float = 0.0
    ) -> np.ndarray:
        """
        Draw gaze estimation results on frame.

        Args:
            frame: Input frame
            results: List of gaze detection results
            processing_time: Processing time in milliseconds

        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()

        # Draw each detection
        for result in results:
            bbox = result["bbox"]
            gaze = result["gaze"]

            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                self.colors["bbox"],
                2,
            )

            # Calculate gaze vector
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            # Convert to radians and calculate endpoint
            pitch_rad = np.radians(gaze["pitch"])
            yaw_rad = np.radians(gaze["yaw"])

            length = 80
            dx = -length * np.sin(pitch_rad) * np.cos(yaw_rad)
            dy = -length * np.sin(yaw_rad)

            end_point = (int(center_x + dx), int(center_y + dy))

            # Draw gaze vector
            cv2.arrowedLine(
                vis_frame,
                (center_x, center_y),
                end_point,
                self.colors["gaze"],
                2,
                tipLength=0.3,
            )

            # Draw gaze angles text
            text = f"P:{gaze['pitch']:.1f}° Y:{gaze['yaw']:.1f}°"
            cv2.putText(
                vis_frame,
                text,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["text"],
                1,
            )

        # Draw performance info
        if processing_time > 0:
            perf_text = f"Processing: {processing_time:.1f}ms"
            cv2.putText(
                vis_frame,
                perf_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return vis_frame


async def real_time_gaze_tracking(
    camera_index: int = 0,
    service_url: str = "http://localhost:8000",
    duration: float = 30.0,
):
    """
    Example: Real-time gaze tracking with performance monitoring.

    Args:
        camera_index: Camera index to use
        service_url: URL of the gaze estimation service
        duration: Duration to run in seconds
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize gaze client and visualizer
    visualizer = GazeVisualizer()

    async with GazeEstimationClient(service_url) as client:
        # Check service health before starting
        if not await client.health_check():
            logger.error("Gaze estimation service is not healthy")
            return

        logger.info("Starting real-time gaze tracking...")
        logger.info(f"Camera: {camera_index}, Duration: {duration}s")
        logger.info("Press 'q' to quit early")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    continue

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                frame_count += 1

                # Process every 3rd frame to reduce load
                if frame_count % 3 == 0:
                    # Get gaze prediction
                    result = await client.predict_frame(frame)

                    if result:
                        results = result.get("results", [])
                        proc_time = result.get("processing_time", 0)

                        # Visualize results
                        vis_frame = visualizer.draw_results(frame, results, proc_time)
                    else:
                        vis_frame = frame
                        # Draw error indicator
                        cv2.putText(
                            vis_frame,
                            "API Error",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                else:
                    vis_frame = frame

                # Add frame counter
                cv2.putText(
                    vis_frame,
                    f"Frame: {frame_count}",
                    (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Display frame
                cv2.imshow("Real-time Gaze Tracking", vis_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

            # Print performance statistics
            stats = client.get_performance_stats()
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            logger.info("Session Summary:")
            logger.info(f"Duration: {elapsed_time:.1f}s")
            logger.info(f"Frames processed: {frame_count}")
            logger.info(f"Average FPS: {fps:.1f}")
            logger.info(f"API requests: {stats['total_requests']}")
            logger.info(f"API errors: {stats['total_errors']}")
            logger.info(f"Error rate: {stats['error_rate']:.1f}%")
            logger.info(f"Avg response time: {stats['avg_response_time']:.1f}ms")


async def batch_image_processing(
    image_paths: List[str], service_url: str = "http://localhost:8000"
):
    """
    Example: Process multiple images in batch with concurrent requests.

    Args:
        image_paths: List of image file paths
        service_url: URL of the gaze estimation service
    """
    logger = logging.getLogger(__name__)

    async with GazeEstimationClient(service_url) as client:
        # Check service health
        if not await client.health_check():
            logger.error("Gaze estimation service is not healthy")
            return

        logger.info(f"Processing {len(image_paths)} images...")

        # Process images concurrently (limit concurrency to avoid overwhelming API)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def process_image(image_path: str):
            async with semaphore:
                try:
                    # Load image
                    frame = cv2.imread(image_path)
                    if frame is None:
                        logger.error(f"Failed to load image: {image_path}")
                        return None

                    # Process image
                    result = await client.predict_frame(frame)
                    if result:
                        logger.info(
                            f"Processed {image_path}: "
                            f"{len(result['results'])} faces, "
                            f"{result['processing_time']:.1f}ms"
                        )
                        return {"path": image_path, "result": result}
                    else:
                        logger.error(f"Failed to process: {image_path}")
                        return None

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    return None

        # Execute batch processing
        start_time = time.time()
        results = await asyncio.gather(
            *[process_image(path) for path in image_paths], return_exceptions=True
        )

        # Filter successful results
        successful_results = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]

        elapsed_time = time.time() - start_time
        logger.info(f"Batch processing completed in {elapsed_time:.1f}s")
        logger.info(f"Successful: {len(successful_results)}/{len(image_paths)}")

        # Print performance stats
        stats = client.get_performance_stats()
        logger.info(
            f"API Performance: {stats['avg_response_time']:.1f}ms avg, {stats['error_rate']:.1f}% errors"
        )

        return successful_results


def main():
    """Main function with example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Gaze Estimation Integration Examples")

    parser.add_argument(
        "--mode",
        choices=["realtime", "batch"],
        default="realtime",
        help="Example mode to run",
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Gaze estimation service URL",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for real-time mode",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration for real-time mode (seconds)",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        help="Image paths for batch mode",
    )

    args = parser.parse_args()

    if args.mode == "realtime":
        print("Starting real-time gaze tracking example...")
        asyncio.run(real_time_gaze_tracking(args.camera, args.url, args.duration))
    elif args.mode == "batch":
        if not args.images:
            print("Please provide image paths with --images for batch mode")
            return
        print("Starting batch processing example...")
        asyncio.run(batch_image_processing(args.images, args.url))


if __name__ == "__main__":
    main()
