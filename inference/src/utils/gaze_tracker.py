from typing import Tuple

import numpy as np


class GazeKalmanTracker:
    """
    Kalman filter for smoothing gaze vector predictions ONLY.

    Should be less restrictive than bbox tracking since gaze can change
    rapidly (e.g., jumping from edge to edge of screen).

    Enabled flag is used for experiment comparing smoothed VS unsmoothed.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize gaze Kalman tracker.

        Args:
            enabled: Whether to apply filtering (set False to disable)
        """
        self.enabled = enabled

        if not enabled:
            return

        # State: [pitch, yaw]
        self.state = None
        self.P = np.eye(2) * 100  # Initial uncertainty
        self.F = np.eye(2)  # State transition
        self.H = np.eye(2)  # Observation matrix

        self.base_Q = np.eye(2) * 0.8  # Higher process noise - gaze changes fast
        self.base_R = (
            np.eye(2) * 2.0
        )  # Lower measurement noise - trust measurements more

        # Current parameters (will be adjusted)
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()

        # Movement threshold in degrees
        self.movement_threshold = 8.0  # degrees

    def update(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """
        Update the Kalman filter with new gaze measurements.

        Args:
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees

        Returns:
            Tuple of (smoothed_pitch, smoothed_yaw) in degrees
        """
        # If disabled, return raw values
        if not self.enabled:
            return pitch, yaw

        z = np.array([pitch, yaw], dtype=np.float32)

        if self.state is None:
            # Initialize with first measurement
            self.state = z
            return pitch, yaw

        # Calculate angular movement magnitude
        movement = np.linalg.norm(z - self.state)

        # Adaptive parameters based on gaze movement
        if movement > self.movement_threshold:
            # Large gaze movement: trust measurements heavily (minimal smoothing)
            self.Q = self.base_Q * 3.0  # Higher process noise
            self.R = self.base_R * 0.3  # Lower measurement noise
        else:
            # Small movement: apply gentle smoothing
            self.Q = self.base_Q * 0.1  # Lower process noise
            self.R = self.base_R * 1.5  # Higher measurement noise

        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = z - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.state[0]), float(self.state[1])

    def reset(self):
        """Reset the tracker state."""
        if not self.enabled:
            return

        self.state = None
        self.P = np.eye(2) * 10.0
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()

    def set_enabled(self, enabled: bool):
        """Enable or disable filtering at runtime."""
        was_enabled = self.enabled
        self.enabled = enabled

        # If switching from disabled to enabled, reset state
        if not was_enabled and enabled:
            self.reset()
