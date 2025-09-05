import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import build_model
from src.datasets import get_dataset
from src.utils.metrics import angular_error_3d
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from tqdm import tqdm
import time

def decode_predictions(predictions, config):
    """Converts binned model output to continuous angular predictions."""
    pitch_pred, yaw_pred = predictions
    device = pitch_pred.device

    num_bins = config['num_bins']
    angle_range = config['angle_range']
    bin_width = angle_range / num_bins

    idx_tensor = torch.arange(num_bins, dtype=torch.float32).to(device)
    pitch_probs = F.softmax(pitch_pred, dim=1)
    yaw_probs = F.softmax(yaw_pred, dim=1)

    pitch_cont = torch.sum(pitch_probs * idx_tensor, 1) * bin_width - (angle_range / 2)
    yaw_cont = torch.sum(yaw_probs * idx_tensor, 1) * bin_width - (angle_range / 2)

    return torch.stack([pitch_cont, yaw_cont], dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_avg_inference_time(model, device):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    for _ in range(10):
        _ = model(dummy_input)

    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    avg_time = (time.time() - start_time) / 100
    return avg_time

def main(cfg_path, weights_path, fused):
    """
    Main evaluation loop for gaze estimation.
    Args:
        cfg_path (str): Path to the training YAML config file.
        weights_path (str): Path to the trained model weights (.pth file).
        fused (bool): If True, builds the model in inference mode for fused weights.
    """
    cfg = load_config(cfg_path)

    logger = get_logger()
    logger.info(f"Evaluating with config: {cfg_path}")
    logger.info(f"Loading weights from: {weights_path}")

    backbone_kwargs = {}
    if fused:
        if cfg['backbone'].startswith('mobileone'):
            logger.info("Preparing to build MobileOne model in INFERENCE MODE.")
            backbone_kwargs['inference_mode'] = True
        else:
            logger.warning(f"The --fused flag is set but the backbone '{cfg['backbone']}' does not support it. Ignoring.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, **backbone_kwargs).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    total_params = count_parameters(model)
    logger.info(f"Total trainable parameters: {total_params}")

    model.eval()

    avg_inference_time = count_avg_inference_time(model, device)
    logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")

    test_dataset = get_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    total_error = 0.0
    total_inference_time = 0.0
    eval_start_time = time.time()

    with torch.no_grad():
        for imgs, _, cont_labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            cont_labels = cont_labels.to(device)

            start_time = time.time()
            predictions = model(imgs)
            total_inference_time += time.time() - start_time

            decoded_preds = decode_predictions(predictions, cfg)
            total_error += angular_error_3d(decoded_preds, cont_labels) * imgs.size(0)

    total_eval_time = time.time() - eval_start_time
    mean_ang_error = total_error / len(test_dataset)
    logger.info("=================================================")
    logger.info(f"Mean Angular Error on the test set: {mean_ang_error:.2f} degrees")
    logger.info(f"Total evaluation time: {total_eval_time:.3f} seconds")
    logger.info(f"Total inference time: {total_inference_time:.3f} seconds")
    logger.info(f"Average inference time per batch: {total_inference_time/len(test_loader):.3f} seconds")
    logger.info("=================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the training config file used.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model weights (.pth).")
    parser.add_argument('--fused', action='store_true', help="Set this flag if evaluating a reparameterized/fused model.")
    args = parser.parse_args()
    main(args.config, args.weights, args.fused)
