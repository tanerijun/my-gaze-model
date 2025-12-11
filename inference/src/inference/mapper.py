from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


class Mapper:
    """Maps gaze vectors to screen coordinates"""

    def __init__(
        self,
        enable_dynamic_calibration: bool = False,
        buffer_size: Optional[int] = None,
    ):
        """
        Initialize the Mapper.

        Args:
            enable_dynamic_calibration: If True, enables dynamic calibration mode
            buffer_size: Maximum number of dynamic calibration samples to keep.
                        If None, keeps all samples (accumulate mode).
                        If set, maintains a fixed-size rolling buffer.
        """
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.is_trained = False

        # Initial calibration data (never modified)
        self.initial_feature_vectors = []
        self.initial_screen_points = []

        # Dynamic calibration data (updated during runtime)
        self.enable_dynamic_calibration = enable_dynamic_calibration
        self.buffer_size = buffer_size
        self.dynamic_feature_vectors = []
        self.dynamic_screen_points = []

        # Statistics tracking
        self.calibration_update_count = 0

    def add_calibration_point(
        self, feature_vectors: List[List[float]], target_point: Tuple[float, float]
    ):
        """
        Add initial calibration data for one target point.

        Args:
            feature_vectors: List of feature vectors from multiple frames. e.g., [pitch, yaw, eye_x, eye_y, ipd, roll]
            target_point: (x, y) screen coordinates for this calibration point
        """
        for feature_vector in feature_vectors:
            self.initial_feature_vectors.append(feature_vector)
            self.initial_screen_points.append(target_point)

    def add_dynamic_calibration_point(
        self, feature_vector: List[float], target_point: Tuple[float, float]
    ):
        """
        Add a single dynamic calibration point

        Args:
            feature_vector: Single feature vector
            target_point: (x, y) screen coordinates
        """
        if not self.enable_dynamic_calibration:
            raise RuntimeError(
                "Dynamic calibration is not enabled. "
                "Initialize Mapper with enable_dynamic_calibration=True"
            )

        self.dynamic_feature_vectors.append(feature_vector)
        self.dynamic_screen_points.append(target_point)

        if (
            self.buffer_size is not None
            and len(self.dynamic_feature_vectors) > self.buffer_size
        ):
            self.dynamic_feature_vectors.pop(0)
            self.dynamic_screen_points.pop(0)

        self.calibration_update_count += 1

    def train(self) -> Tuple[float, float]:
        """
        Train the mapping models using collected calibration data.

        Returns:
            Tuple[float, float]: R² scores for x and y coordinates
        """
        # Copy to avoid mutating initial data
        all_feature_vectors = self.initial_feature_vectors.copy()
        all_screen_points = self.initial_screen_points.copy()

        if self.enable_dynamic_calibration:
            all_feature_vectors.extend(self.dynamic_feature_vectors)
            all_screen_points.extend(self.dynamic_screen_points)

        if len(all_feature_vectors) == 0:
            raise ValueError("No calibration data available for training")

        X = np.array(all_feature_vectors)
        screen_points = np.array(all_screen_points)

        y_x = screen_points[:, 0]  # x coordinates
        y_y = screen_points[:, 1]  # y coordinates

        # Train separate models for x and y
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self.is_trained = True

        # Return R² scores
        score_x = float(self.model_x.score(X, y_x))
        score_y = float(self.model_y.score(X, y_y))

        return score_x, score_y

    def predict(self, gaze_vector: List[float]) -> Tuple[float, float]:
        """
        Map gaze vector to screen coordinates.

        Args:
            feature_vector: A list of features (e.g., [pitch, yaw, ...])

        Returns:
            Tuple[float, float]: Predicted screen coordinates (x, y)
        """
        if not self.is_trained:
            raise ValueError("Mapper must be trained before prediction")

        X = np.array(gaze_vector).reshape(1, -1)
        screen_x = float(self.model_x.predict(X)[0])
        screen_y = float(self.model_y.predict(X)[0])
        return screen_x, screen_y

    def reset(self):
        """Reset all training data and models."""
        self.initial_feature_vectors = []
        self.initial_screen_points = []
        self.dynamic_feature_vectors = []
        self.dynamic_screen_points = []
        self.is_trained = False
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.calibration_update_count = 0

    def reset_dynamic_calibration(self):
        """Reset only the dynamic calibration data, preserving initial calibration."""
        self.dynamic_feature_vectors = []
        self.dynamic_screen_points = []
        self.calibration_update_count = 0
        # Retrain with only initial calibration data
        if len(self.initial_feature_vectors) > 0:
            self.train()

    def get_training_stats(self) -> dict:
        """Get statistics about training data."""
        return {
            "initial_samples": len(self.initial_feature_vectors),
            "dynamic_samples": len(self.dynamic_feature_vectors),
            "total_samples": len(self.initial_feature_vectors)
            + len(self.dynamic_feature_vectors),
            "is_trained": self.is_trained,
            "dynamic_calibration_enabled": self.enable_dynamic_calibration,
            "buffer_size": self.buffer_size,
            "calibration_updates": self.calibration_update_count,
        }
