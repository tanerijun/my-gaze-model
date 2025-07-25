import yaml
import argparse
import torch
import os
from src.models import build_model
from src.utils.logger import get_logger
import inspect

def reparameterize_model(model: torch.nn.Module):
    """
    Iterates through all modules of a model and calls the reparameterize()
    method if it exists. This is used to fuse the branches of MobileOne blocks.

    Args:
        model (torch.nn.Module): The model to reparameterize.

    Returns:
        torch.nn.Module: The reparameterized model.
    """
    for module in model.modules():
        if hasattr(module, 'reparameterize') and inspect.ismethod(module.reparameterize):
            module.reparameterize()
    return model

def main(cfg_path, weights_path, output_path):
    """
    Loads a trained model, reparameterizes it for inference, and saves the new weights.

    Args:
        cfg_path (str): Path to the training YAML config file.
        weights_path (str): Path to the trained (unfused) model weights.
        output_path (str): Path to save the new fused model weights.
    """
    logger = get_logger()

    # --- 1. Load Configuration ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if not cfg['backbone'].startswith('mobileone'):
        logger.error("Error: Reparameterization is only applicable to MobileOne backbones.")
        return

    logger.info(f"Loading model with config: {cfg_path}")

    # --- 2. Build and Load Unfused Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info(f"Successfully loaded unfused weights from: {weights_path}")

    # --- 3. Reparameterize the Model ---
    logger.info("Fusing model branches for inference...")
    reparameterized_model = reparameterize_model(model)
    logger.info("Model reparameterization complete.")

    # --- 4. Save Fused Model ---
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(reparameterized_model.state_dict(), output_path)
    logger.info(f"Fused inference-ready model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileOne Model Reparameterization")
    parser.add_argument('--config', type=str, required=True, help="Path to the MobileOne training config file.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained (unfused) model weights.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the fused model weights.")
    args = parser.parse_args()
    main(args.config, args.weights, args.output)
