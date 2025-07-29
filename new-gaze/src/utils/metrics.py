import numpy as np

def angular_error_3d(pred_angles, gt_angles):
    """
    Computes the mean 3D angular error between predicted and target gaze vectors.
    This is the correct metric for evaluating gaze estimation.

    Args:
        pred_angles (Tensor): Predicted pitch and yaw angles, shape [B, 2] (in degrees).
        gt_angles (Tensor): Ground truth pitch and yaw angles, shape [B, 2] (in degrees).
    Returns:
        float: Mean angular error in degrees.
    """
    # Convert predictions and ground truth to numpy for processing
    pred_angles = pred_angles.cpu().numpy()
    gt_angles = gt_angles.cpu().numpy()

    # Convert pitch and yaw to 3D gaze vectors
    def to_3d_vector(angles):
        # angles are [pitch, yaw] in degrees
        pitch = np.deg2rad(angles[:, 0])
        yaw = np.deg2rad(angles[:, 1])

        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)

        # Stack to create [B, 3] vectors and normalize
        vectors = np.stack([x, y, z], axis=1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    pred_vecs = to_3d_vector(pred_angles)
    gt_vecs = to_3d_vector(gt_angles)

    # Calculate dot product
    dot_product = np.sum(pred_vecs * gt_vecs, axis=1)

    # Clip to avoid arccos domain errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angular error in degrees
    angular_diff = np.arccos(dot_product) * 180.0 / np.pi

    return np.mean(angular_diff)
