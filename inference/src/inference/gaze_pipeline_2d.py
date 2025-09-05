from typing import Optional, Tuple

import numpy as np

from .gaze_pipeline_3d import GazePipeline3D
from .mapper import Mapper


class GazePipeline2D:
    """2D Gaze Pipeline that uses a trained Mapper and 3D pipeline for POG inference."""

    def __init__(self, pipeline_3d: GazePipeline3D, mapper: Mapper):
        """
        Initialize 2D pipeline.

        Args:
            pipeline_3d: Trained 3D gaze estimation pipeline
            mapper: Trained gaze mapper (must be trained before use)
        """
        self.pipeline_3d = pipeline_3d
        self.mapper = mapper

    def predict(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Predict Point of Gaze (POG) from frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            Optional[Tuple[float, float]]: Predicted screen coordinates (x, y) or None if no face detected
        """
        if not self.mapper.is_trained:
            raise ValueError("Mapper must be trained before inference")

        # Get 3D gaze vector
        results = self.pipeline_3d(frame)
        if not results:
            return None

        # Use first detected face
        gaze = results[0]["gaze"]
        gaze_vector = [gaze["pitch"], gaze["yaw"]]

        # Map to screen coordinates
        return self.mapper.predict(gaze_vector)

    def reset_tracking(self):
        """Reset 3D pipeline tracking state."""
        self.pipeline_3d.reset_tracking()
