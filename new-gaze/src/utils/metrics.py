import torch
import math

def angular_error(pred, target):
    """
    Computes mean angular error (in degrees) between predicted and target gaze vectors.
    Args:
        pred (Tensor): Predicted gaze vectors, shape [B, 3] or [B, 2].
        target (Tensor): Ground truth gaze vectors, same shape as pred.
    Returns:
        float: Mean angular error in degrees.
    """
    pred = pred / pred.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    dot = (pred * target).sum(dim=-1).clamp(-1, 1)
    error = torch.acos(dot) * 180.0 / math.pi
    return error.mean().item()
