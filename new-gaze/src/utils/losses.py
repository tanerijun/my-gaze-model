import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeLoss(nn.Module):
    """
    Computes the hybrid gaze loss, combining classification and regression.
    """
    def __init__(self, config):
        super().__init__()
        self.num_bins = config['num_bins']
        self.angle_range = config['angle_range']
        self.bin_width = config['bin_width']
        self.alpha = config.get('alpha', 1.0) # weight for the regression loss

        # Loss functions
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

        # Create a tensor of bin indices [0, 1, 2, ..., num_bins-1]
        self.idx_tensor = torch.arange(self.num_bins, dtype=torch.float32)

    def forward(self, predictions, binned_labels, cont_labels):
        """
        Args:
            predictions (tuple): A tuple containing (pitch_logits, yaw_logits)
                                 from the model. Each has shape [B, num_bins].
            binned_labels (Tensor): Ground truth binned labels, shape [B, 2].
            cont_labels (Tensor): Ground truth continuous labels, shape [B, 2].
        Returns:
            Tensor: The combined scalar loss value.
        """
        pitch_pred, yaw_pred = predictions
        pitch_binned_gt, yaw_binned_gt = binned_labels[:, 0], binned_labels[:, 1]
        pitch_cont_gt, yaw_cont_gt = cont_labels[:, 0], cont_labels[:, 1]

        # Ensure index tensor is on the same device as predictions
        self.idx_tensor = self.idx_tensor.to(pitch_pred.device)

        # --- Classification Loss ---
        loss_pitch_cls = self.cls_loss(pitch_pred, pitch_binned_gt)
        loss_yaw_cls = self.cls_loss(yaw_pred, yaw_binned_gt)

        # --- Regression Loss ---
        # Apply softmax to get probabilities from logits
        pitch_probs = F.softmax(pitch_pred, dim=1)
        yaw_probs = F.softmax(yaw_pred, dim=1)

        # Calculate expected value of bins
        pitch_cont_pred = torch.sum(pitch_probs * self.idx_tensor, 1) * self.bin_width - (self.angle_range / 2)
        yaw_cont_pred = torch.sum(yaw_probs * self.idx_tensor, 1) * self.bin_width - (self.angle_range / 2)

        # Calculate regression loss
        loss_pitch_reg = self.reg_loss(pitch_cont_pred, pitch_cont_gt)
        loss_yaw_reg = self.reg_loss(yaw_cont_pred, yaw_cont_gt)

        # --- Combine Losses ---
        total_loss = (loss_pitch_cls + loss_yaw_cls) + self.alpha * (loss_pitch_reg + loss_yaw_reg)

        return total_loss
