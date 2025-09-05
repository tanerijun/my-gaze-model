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
        self.training_gaze_vectors = []
        self.training_screen_points = []

    def add_calibration_point(
        self, gaze_vectors: List[List[float]], target_point: Tuple[float, float]
    ):
        """
        Add calibration data for one target point.

        Args:
            gaze_vectors: List of [pitch, yaw] vectors from multiple frames
            target_point: (x, y) screen coordinates for this calibration point
        """
        for gaze_vector in gaze_vectors:
            self.training_gaze_vectors.append(gaze_vector)
            self.training_screen_points.append(target_point)

    def train(self) -> Tuple[float, float]:
        """
        Train the mapping models using collected calibration data.

        Returns:
            Tuple[float, float]: R² scores for x and y coordinates
        """
        if len(self.training_gaze_vectors) == 0:
            raise ValueError("No calibration data available for training")

        X = np.array(self.training_gaze_vectors)  # [N, 2] - pitch, yaw
        screen_points = np.array(self.training_screen_points)  # [N, 2] - x, y

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
            gaze_vector: [pitch, yaw] in degrees

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
        self.training_gaze_vectors = []
        self.training_screen_points = []
        self.is_trained = False
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()

    def get_training_stats(self) -> dict:
        """Get statistics about training data."""
        return {
            "num_samples": len(self.training_gaze_vectors),
            "is_trained": self.is_trained,
        }
