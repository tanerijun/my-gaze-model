import numpy as np


class KalmanBoxTracker:
    """
    Kalman filter for smoothing bounding box predictions ONLY.
    Provides adaptive noise parameters based on movement detection.
    """

    def __init__(self):
        # State: [x1, y1, x2, y2]
        self.state = None
        self.P = np.eye(4) * 100  # initial uncertainty
        self.F = np.eye(4)  # state transition matrix
        self.H = np.eye(4)  # observation matrix

        # Base parameters - tuned for stability
        self.base_Q = np.eye(4) * 0.05  # process noise
        self.base_R = np.eye(4) * 25.0  # measurement noise

        # Current parameters (will be adjusted dynamically)
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()

        self.movement_threshold = 30  # pixels

    def update(self, bbox):
        """
        Update the Kalman filter with a new bounding box observation.

        Args:
            bbox: List or tuple of [x1, y1, x2, y2] coordinates

        Returns:
            list: Smoothed bounding box coordinates [x1, y1, x2, y2]
        """
        z = np.array(bbox, dtype=np.float32)

        if self.state is None:
            # Initialize state with first observation
            self.state = z
            return bbox

        # Calculate movement magnitude
        movement = np.linalg.norm(z - self.state)

        # Adaptive parameters based on movement
        if movement > self.movement_threshold:
            # High movement: trust measurements more
            self.Q = self.base_Q * 20.0
            self.R = self.base_R * 0.1
        else:
            # Low movement: prioritize stability
            self.Q = self.base_Q * 0.2
            self.R = self.base_R * 2.0

        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = z - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return [int(x) for x in self.state]

    def reset(self):
        """Reset the tracker state."""
        self.state = None
        self.P = np.eye(4) * 100
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()
