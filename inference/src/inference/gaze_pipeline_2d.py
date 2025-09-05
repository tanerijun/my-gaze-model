from typing import Dict, List

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

    def predict(self, frame: np.ndarray) -> List[Dict]:
        """
        Predict Point of Gaze (POG) from frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            List[Dict]: Same format as 3D pipeline but with added "pog" field:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "gaze": {"pitch": float, "yaw": float},
                    "pog": {"x": float, "y": float}  # Added POG coordinates
                }
            ]
        """
        if not self.mapper.is_trained:
            raise ValueError("Mapper must be trained before inference")

        # Get 3D gaze results
        results_3d = self.pipeline_3d(frame)

        # Augment each result with POG
        results_2d = []
        for result in results_3d:
            gaze = result["gaze"]
            gaze_vector = [gaze["pitch"], gaze["yaw"]]

            # Map to screen coordinates
            pog_coords = self.mapper.predict(gaze_vector)

            # Create augmented result
            result_2d = result.copy()
            result_2d["pog"] = {"x": pog_coords[0], "y": pog_coords[1]}
            results_2d.append(result_2d)

        return results_2d

    def reset_tracking(self):
        """Reset 3D pipeline tracking state."""
        self.pipeline_3d.reset_tracking()
