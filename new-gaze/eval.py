import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import build_model
from src.datasets import get_dataset
from src.utils.metrics import angular_error
from src.utils.logger import get_logger
from tqdm import tqdm

def decode_predictions(predictions, config):
    """Converts binned model output to continuous angular predictions."""
    pitch_pred, yaw_pred = predictions

    # Get device from one of the tensors
    device = pitch_pred.device

    # Get config values
    num_bins = config['num_bins']
    angle_range = config['angle_range']
    bin_width = config['bin_width']

    # Create and move index tensor to the correct device
    idx_tensor = torch.arange(num_bins, dtype=torch.float32).to(device)

    # Apply softmax to get probabilities
    pitch_probs = F.softmax(pitch_pred, dim=1)
    yaw_probs = F.softmax(yaw_pred, dim=1)

    # Calculate expected value (continuous angle)
    pitch_cont = torch.sum(pitch_probs * idx_tensor, 1) * bin_width - (angle_range / 2)
    yaw_cont = torch.sum(yaw_probs * idx_tensor, 1) * bin_width - (angle_range / 2)

    # Stack them into a [B, 2] tensor
    return torch.stack([pitch_cont, yaw_cont], dim=1)

def main(cfg_path, weights_path):
    """
    Main evaluation loop for gaze estimation.
    Args:
        cfg_path (str): Path to the training YAML config file.
        weights_path (str): Path to the trained model weights (.pth file).
    """
    # --- Load Configuration and Logger ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    logger = get_logger()
    logger.info(f"Evaluating with config: {cfg_path}")
    logger.info(f"Loading weights from: {weights_path}")

    # --- Setup Device and Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Load Test Dataset ---
    cfg['split'] = 'test'
    test_dataset = get_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Evaluation Loop ---
    total_error = 0.0
    with torch.no_grad():
        for imgs, _, cont_labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            cont_labels = cont_labels.to(device) # Ground truth [pitch, yaw] in degrees

            predictions = model(imgs)

            # Decode the binned output to continuous angles
            decoded_preds = decode_predictions(predictions, cfg)

            # Calculate angular error
            total_error += angular_error(decoded_preds, cont_labels) * imgs.size(0)

    mean_ang_error = total_error / len(test_dataset)
    logger.info("=================================================")
    logger.info(f"Mean Angular Error on the test set: {mean_ang_error:.2f} degrees")
    logger.info("=================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the training config file used.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights (.pth).")
    args = parser.parse_args()
    main(args.config, args.weights)
