"""
WebSocket Service
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import base64
import json
import logging
import queue
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
import websockets

from src import GazePipeline3D


class GazeService:
    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        smooth_gaze: bool = True,
        port: int = 8765,
    ):
        """
        Initialize the gaze estimation service.

        Args:
            weights_path: Path to the trained model weights
            device: Compute device ("cpu", "cuda", or "auto")
            smooth_gaze: Enable Kalman filtering for gaze vectors
            port: WebSocket server port
        """
        self.port = port
        self.pipeline = None
        self.weights_path = weights_path
        self.device = device
        self.smooth_gaze = smooth_gaze

        # WebSocket connections
        self.clients = set()

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.running = False

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting gaze service on port {self.port}")

        # Initialize gaze pipeline
        self.logger.info("Initializing gaze pipeline...")
        self.pipeline = GazePipeline3D(
            self.weights_path, device=self.device, smooth_gaze=self.smooth_gaze
        )

        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.start()

        # Start WebSocket server
        async with websockets.serve(
            self.handle_client,
            "localhost",
            self.port,
        ):
            self.logger.info(f"Gaze service running on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever

    def stop_server(self):
        """Stop the service."""
        self.logger.info("Stopping gaze service...")
        self.running = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

    async def handle_client(self, websocket, path=None):
        """Handle new WebSocket connections."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        self.logger.info(f"Client connected: {client_addr}")

        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            self.logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            self.clients.discard(websocket)

    async def process_message(self, websocket, message: str):
        """Process incoming messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "frame":
                await self.handle_frame(data)
            elif message_type == "reset_tracking":
                self.reset_tracking()
                await websocket.send(
                    json.dumps({"type": "tracking_reset", "success": True})
                )
            elif message_type == "get_status":
                await self.send_status(websocket)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            self.logger.error("Invalid JSON received")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def handle_frame(self, data: Dict):
        """Handle incoming frame data."""
        try:
            # Decode base64 frame
            frame_data = data.get("frame")
            if not frame_data:
                return

            # Remove data URL prefix if present
            if frame_data.startswith("data:image"):
                frame_data = frame_data.split(",")[1]

            # Decode image
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                self.logger.error("Failed to decode frame")
                return

            # Add frame to processing queue (non-blocking)
            try:
                self.frame_queue.put_nowait(
                    {
                        "frame": frame,
                        "timestamp": time.time(),
                        "frame_id": data.get("frame_id", 0),
                    }
                )
            except queue.Full:
                # Skip frame if queue is full to maintain low latency
                pass

        except Exception as e:
            self.logger.error(f"Error handling frame: {e}")

    def _process_frames(self):
        """Background thread for processing frames."""
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=0.1)

                # Process frame
                start_time = time.time()
                if self.pipeline is not None:
                    results = self.pipeline(frame_data["frame"])
                else:
                    results = []
                processing_time = time.time() - start_time

                # Send results to all connected clients
                response = {
                    "type": "gaze_results",
                    "frame_id": frame_data.get("frame_id", 0),
                    "timestamp": frame_data["timestamp"],
                    "processing_time": processing_time * 1000,  # ms
                    "results": results,
                }

                # Broadcast to all clients
                try:
                    loop = asyncio.get_event_loop()
                    if loop and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_results(response), loop
                        )
                except RuntimeError:
                    # Handle case where no event loop exists
                    pass

                # Update performance stats
                self.frame_count += 1

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")

    async def _broadcast_results(self, response: Dict):
        """Broadcast results to all connected clients."""
        if not self.clients:
            return

        message = json.dumps(response)
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                self.logger.error(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    def reset_tracking(self):
        """Reset tracking state."""
        if self.pipeline:
            self.pipeline.reset_tracking()
            self.logger.info("Tracking state reset")

    async def send_status(self, websocket):
        """Send service status to client."""
        uptime = time.time() - self.start_time
        avg_fps = self.frame_count / uptime if uptime > 0 else 0

        status = {
            "type": "status",
            "uptime": uptime,
            "frames_processed": self.frame_count,
            "average_fps": avg_fps,
            "device": str(self.pipeline.device) if self.pipeline else "not initialized",
            "smooth_gaze": self.smooth_gaze,
            "connected_clients": len(self.clients),
        }

        await websocket.send(json.dumps(status))


class CameraFrameProducer:
    """Helper class for camera-based testing."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if ret:
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            return frame
        return None

    def stop(self):
        if self.cap:
            self.cap.release()


async def test_camera_mode(service: GazeService):
    """Test mode using local camera."""
    camera = CameraFrameProducer()

    try:
        camera.start()
        print("Camera test mode - press 'q' to quit")

        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # Process directly for testing
            if service.pipeline:
                results = service.pipeline(frame)

                # Draw results
                for result in results:
                    bbox = result["bbox"]
                    gaze = result["gaze"]

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                    )

                    # Draw gaze vector
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)

                    pitch_rad = np.radians(gaze["pitch"])
                    yaw_rad = np.radians(gaze["yaw"])

                    length = 100
                    dx = -length * np.sin(pitch_rad) * np.cos(yaw_rad)
                    dy = -length * np.sin(yaw_rad)

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

            # Display frame
            cv2.imshow("Gaze Service Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Gaze Estimation WebSocket Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gaze_service.py --weights model.pth                   # Start service
  python gaze_service.py --weights model.pth --port 9000       # Custom port
  python gaze_service.py --weights model.pth --device cpu      # Force CPU
  python gaze_service.py --weights model.pth --test-camera     # Camera test mode
        """,
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth file)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
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
        default=True,
        help="Enable Kalman filtering for gaze vectors (default: True)",
    )

    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Test mode using local camera (for debugging)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Gaze Estimation WebSocket Service")
    print("=" * 60)
    print(f"Model weights: {args.weights}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device}")
    print(f"Gaze smoothing: {args.smooth_gaze}")
    if args.test_camera:
        print("Mode: Camera test")
    else:
        print("Mode: WebSocket service")
    print("=" * 60)

    service = GazeService(
        weights_path=args.weights,
        device=args.device,
        smooth_gaze=args.smooth_gaze,
        port=args.port,
    )

    try:
        if args.test_camera:
            # Initialize pipeline for testing
            service.pipeline = GazePipeline3D(
                args.weights, device=args.device, smooth_gaze=args.smooth_gaze
            )
            asyncio.run(test_camera_mode(service))
        else:
            # Start WebSocket service
            asyncio.run(service.start_server())
    except KeyboardInterrupt:
        print("\nService interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        service.stop_server()


if __name__ == "__main__":
    main()
