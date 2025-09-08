from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


class Mapper:
    """Maps gaze vectors to screen coordinates"""

    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.is_trained = False

        # Store training data for inspection/retraining
        self.training_feature_vectors = []
        self.training_screen_points = []

    def add_calibration_point(
        self, feature_vectors: List[List[float]], target_point: Tuple[float, float]
    ):
        """
        Add calibration data for one target point.

        Args:
            feature_vectors: List of feature vectors from multiple frames. e.g., [pitch, yaw, eye_x, eye_y, ipd, roll]
            target_point: (x, y) screen coordinates for this calibration point
        """
        for feature_vector in feature_vectors:
            self.training_feature_vectors.append(feature_vector)
            self.training_screen_points.append(target_point)

    def train(self) -> Tuple[float, float]:
        """
        Train the mapping models using collected calibration data.

        Returns:
            Tuple[float, float]: R² scores for x and y coordinates
        """
        if len(self.training_feature_vectors) == 0:
            raise ValueError("No calibration data available for training")

        X = np.array(self.training_feature_vectors)
        screen_points = np.array(self.training_screen_points)

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
        self.training_feature_vectors = []
        self.training_screen_points = []
        self.is_trained = False
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()

    def get_training_stats(self) -> dict:
        """Get statistics about training data."""
        return {
            "num_samples": len(self.training_feature_vectors),
            "is_trained": self.is_trained,
        }
