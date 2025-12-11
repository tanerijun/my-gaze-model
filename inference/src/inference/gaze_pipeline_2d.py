from typing import Dict, List

import numpy as np

from .gaze_pipeline_3d import GazePipeline3D
from .mapper import Mapper


class GazePipeline2D:
    """2D Gaze Pipeline that encapsulates the logic for mapping 3D gaze data to 2D screen coordinates"""

    def __init__(
        self,
        pipeline_3d: GazePipeline3D,
        mapper: Mapper,
        feature_keys: List[str],
    ):
        """
        Initialize 2D pipeline.

        Args:
            pipeline_3d: An initialized 3D gaze estimation pipeline.
            mapper: An external Mapper instance that will be used for training and prediction.
            feature_keys: A list of strings defining the features to use for mapping.
        """
        self.pipeline_3d = pipeline_3d
        self.mapper = mapper  # Uses an externally provided mapper

        if not isinstance(feature_keys, list) or not feature_keys:
            raise ValueError("feature_keys must be a non-empty list of strings.")
        self.feature_keys = feature_keys
        print(
            f"GazePipeline2D initialized to use {len(self.feature_keys)} features: {self.feature_keys}"
        )

    @property
    def is_calibrated(self) -> bool:
        """Returns True if the internal mapper has been trained."""
        return self.mapper.is_trained

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
        if not self.is_calibrated:
            raise RuntimeError("Mapper must be trained before calling predict.")

        h, w = frame.shape[:2]

        results_3d = self.pipeline_3d(frame)
        results_2d = []
        for result in results_3d:
            try:
                feature_vector = self.extract_feature_vector(result)
                pog_coords = self.mapper.predict(feature_vector)
            except ValueError as e:
                # If a feature is missing, we can't predict. Skip this face.
                print(f"Warning: Could not predict for a face. Error: {e}")
                continue

            pog_x = pog_coords[0]
            pog_y = pog_coords[1]

            # Create augmented result
            result_2d = result.copy()
            result_2d["pog"] = {"x": pog_x, "y": pog_y}
            results_2d.append(result_2d)

        return results_2d

    def reset_tracking(self):
        """Reset 3D pipeline tracking state."""
        self.pipeline_3d.reset_tracking()

    def extract_feature_vector(self, result_3d: Dict) -> List[float]:
        """
        Strictly extracts a configured feature vector from a 3D pipeline result.
        This is the primary utility for both calibration and prediction.

        Args:
            result_3d: The dictionary output from GazePipeline3D for one person.

        Returns:
            A list of float values corresponding to the instance's feature_keys.
        """
        flat_features = {}
        if "gaze" in result_3d:
            flat_features.update(result_3d["gaze"])
        if "gaze_origin_features" in result_3d:
            flat_features.update(result_3d["gaze_origin_features"])

        feature_vector = []
        for key in self.feature_keys:
            if key not in flat_features:
                raise ValueError(
                    f"Required feature key '{key}' not found in the 3D pipeline output. "
                    f"Available keys are: {list(flat_features.keys())}"
                )
            feature_vector.append(flat_features[key])

        return feature_vector
