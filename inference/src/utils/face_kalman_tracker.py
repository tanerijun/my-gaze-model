import numpy as np


class FaceKalmanTracker:
    """
    Kalman filter for smoothing both bounding box and keypoints predictions.
    It maintains a unified state for bbox [x1, y1, x2, y2] and
    6 keypoints [(x, y) * 6].

    The adaptive noise parameters are adjusted based on the movement of the
    bounding box only, as per the project requirements.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return

        self.state_dim = 16  # 4 for bbox + 12 for 6 keypoints
        self.state = None
        self.P = np.eye(self.state_dim) * 100

        self.F = np.eye(self.state_dim)
        self.H = np.eye(self.state_dim)

        q_bbox_base = 0.05  # Low process noise = expect smooth movement
        r_bbox_base = (
            25.0  # High measurement noise = less trust in detector for stability
        )

        q_kps_base = 0.8  # Higher process noise = expect more jitter/movement
        r_kps_base = 4.0  # Lower measurement noise = trust detector's keypoints more

        q_diag = [q_bbox_base] * 4 + [q_kps_base] * 12
        self.base_Q = np.diag(q_diag)

        r_diag = [r_bbox_base] * 4 + [r_kps_base] * 12
        self.base_R = np.diag(r_diag)

        # Current parameters (will be adjusted dynamically)
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()

        # Movement threshold for bbox only (in pixels)
        self.movement_threshold = 30

    def update(self, measurement):
        """
        Update the Kalman filter with a new observation.

        Args:
            measurement: A list or array of 16 values:
                            [x1, y1, x2, y2, kp0_x, kp0_y, ..., kp5_x, kp5_y]

        Returns:
            list: Smoothed state vector of 16 values.
        """
        if not self.enabled:
            # If not enabled, return integer bbox coords and float keypoint coords
            int_bbox = [int(x) for x in measurement[:4]]
            return int_bbox + measurement[4:]

        z = np.array(measurement, dtype=np.float32)

        if self.state is None:
            self.state = z
            int_bbox = [int(x) for x in z[:4]]
            return int_bbox + z[4:].tolist()

        # Calculate movement magnitude using ONLY the bbox part of the state
        bbox_movement = np.linalg.norm(z[:4] - self.state[:4])

        # Adaptive parameters based on bbox movement (using original multipliers)
        if bbox_movement > self.movement_threshold:
            # High movement: trust all measurements more
            self.Q = self.base_Q * 20.0
            self.R = self.base_R * 0.1
        else:
            # Low movement: prioritize stability for all state variables
            self.Q = self.base_Q * 0.2
            self.R = self.base_R * 2.0

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        smoothed_bbox = [int(x) for x in self.state[:4]]
        smoothed_kps = self.state[4:].tolist()

        return smoothed_bbox + smoothed_kps

    def reset(self):
        """Reset the tracker state."""
        if not self.enabled:
            return

        self.state = None
        self.P = np.eye(self.state_dim) * 100
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()
