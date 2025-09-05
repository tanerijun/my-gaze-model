"""
REST API Service for Gaze Estimation
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import base64
import logging
import time
from typing import Dict, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import GazePipeline3D


class GazeResult(BaseModel):
    bbox: List[int]  # [x1, y1, x2, y2]
    gaze: Dict[str, float]  # {"pitch": float, "yaw": float}


class GazeResponse(BaseModel):
    results: List[GazeResult]
    processing_time: float  # milliseconds
    timestamp: float


class StatusResponse(BaseModel):
    status: str
    uptime: float
    frames_processed: int
    average_fps: float
    device: str
    smooth_gaze: bool


class GazeRestService:
    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        smooth_gaze: bool = False,  # not really suitable for REST since request might not be sequential
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Initialize the REST API service for gaze estimation.

        Args:
            weights_path: Path to the trained model weights
            device: Compute device ("cpu", "cuda", or "auto")
            smooth_gaze: Enable Kalman filtering for gaze vectors
            host: Host address to bind to
            port: Port to run the service on
        """
        self.weights_path = weights_path
        self.device = device
        self.smooth_gaze = smooth_gaze
        self.host = host
        self.port = port

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        # Initialize pipeline
        self.pipeline = GazePipeline3D(
            self.weights_path, device=self.device, smooth_gaze=self.smooth_gaze
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Gaze Estimation API",
            description="REST API for real-time gaze estimation",
            version="1.0.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routes
        app.post("/predict", response_model=GazeResponse)(self.predict)
        app.post("/predict_base64", response_model=GazeResponse)(self.predict_base64)
        app.get("/status", response_model=StatusResponse)(self.get_status)
        app.post("/reset")(self.reset_tracking)
        app.get("/health")(self.health_check)

        return app

    async def predict(self, file: UploadFile = File(...)) -> GazeResponse:
        """
        Process uploaded image file for gaze estimation.

        Args:
            file: Uploaded image file

        Returns:
            GazeResponse with detected faces and gaze directions
        """
        try:
            # Read and decode image
            contents = await file.read()
            image = self._decode_image_from_bytes(contents)

            # Process image
            start_time = time.time()

            if not self.pipeline:
                raise ValueError("pipeline is supposed to be available here")

            self.pipeline.reset_tracking()  # requests might not be sequential

            results = self.pipeline(image)
            processing_time = (time.time() - start_time) * 1000  # convert to ms

            # Update stats
            self.frame_count += 1

            # Format results
            formatted_results = [
                GazeResult(bbox=result["bbox"], gaze=result["gaze"])
                for result in results
            ]

            return GazeResponse(
                results=formatted_results,
                processing_time=processing_time,
                timestamp=time.time(),
            )

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise HTTPException(
                status_code=400, detail=f"Error processing image: {str(e)}"
            )

    async def predict_base64(self, request: Dict) -> GazeResponse:
        """
        Process base64 encoded image for gaze estimation.

        Args:
            request: Dictionary containing base64 encoded image

        Returns:
            GazeResponse with detected faces and gaze directions
        """
        try:
            # Extract base64 image data
            image_data = request.get("image")
            if not image_data:
                raise HTTPException(
                    status_code=400, detail="Missing 'image' field in request"
                )

            # Remove data URL prefix if present
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]

            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = self._decode_image_from_bytes(image_bytes)

            # Process image
            start_time = time.time()

            if not self.pipeline:
                raise ValueError("pipeline is supposed to exist here")

            results = self.pipeline(image)
            processing_time = (time.time() - start_time) * 1000  # convert to ms

            # Update stats
            self.frame_count += 1

            # Format results
            formatted_results = [
                GazeResult(bbox=result["bbox"], gaze=result["gaze"])
                for result in results
            ]

            return GazeResponse(
                results=formatted_results,
                processing_time=processing_time,
                timestamp=time.time(),
            )

        except Exception as e:
            self.logger.error(f"Error processing base64 image: {e}")
            raise HTTPException(
                status_code=400, detail=f"Error processing image: {str(e)}"
            )

    async def get_status(self) -> StatusResponse:
        """Get service status and statistics."""
        uptime = time.time() - self.start_time
        avg_fps = self.frame_count / uptime if uptime > 0 else 0

        return StatusResponse(
            status="running" if self.pipeline else "initializing",
            uptime=uptime,
            frames_processed=self.frame_count,
            average_fps=avg_fps,
            device=str(self.pipeline.device) if self.pipeline else "not initialized",
            smooth_gaze=self.smooth_gaze,
        )

    async def reset_tracking(self):
        """Reset tracking state."""
        if self.pipeline:
            self.pipeline.reset_tracking()
            self.logger.info("Tracking state reset")
            return {"message": "Tracking reset successfully"}
        else:
            raise HTTPException(status_code=400, detail="Pipeline not initialized")

    async def health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "pipeline_initialized": self.pipeline is not None,
        }

    def _decode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode image from bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Decoded image as numpy array in BGR format
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            return image

        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")

    def run(self):
        """Start the REST API server."""
        self.logger.info("Starting gaze estimation REST API server...")
        self.logger.info(f"Host: {self.host}, Port: {self.port}")
        self.logger.info(f"Model weights: {self.weights_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Gaze smoothing: {self.smooth_gaze}")

        # Run server
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Gaze Estimation REST API Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rest_service.py --weights model.pth                     # Start service on default port
  python rest_service.py --weights model.pth --port 8080         # Custom port
  python rest_service.py --weights model.pth --device cpu        # Force CPU
  python rest_service.py --weights model.pth --host 0.0.0.0      # Bind to all interfaces
        """,
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth file)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the service on (default: 8000)",
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
        default=False,
        help="Enable Kalman filtering for gaze vectors (default: False)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Gaze Estimation REST API Service")
    print("=" * 60)
    print(f"Model weights: {args.weights}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device}")
    print(f"Gaze smoothing: {args.smooth_gaze}")
    print("=" * 60)

    # Create and run service
    service = GazeRestService(
        weights_path=args.weights,
        device=args.device,
        smooth_gaze=args.smooth_gaze,
        host=args.host,
        port=args.port,
    )

    try:
        service.run()
    except KeyboardInterrupt:
        print("\nService interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
