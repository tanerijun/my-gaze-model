import yaml
import argparse
import os
import torch
from torch.utils.data import DataLoader
from src.models import build_model
from src.datasets import get_dataset
from src.utils.losses import GazeLoss
from src.utils.logger import get_logger
import time

def main(cfg_path):
    """
    Main training loop for gaze estimation.
    Args:
        cfg_path (str): Path to YAML config file.
    """
    # --- Load Configuration and Logger ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('output', f"{cfg['dataset_name']}_{cfg['backbone']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, 'train.log')
    logger = get_logger(log_file_path)
    logger.info("--- Training Configuration ---")
    for key, val in cfg.items():
        logger.info(f"{key}: {val}")
    logger.info("-----------------------------")

    # --- Setup Device, Model, and Loss ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = build_model(cfg).to(device)
    criterion = GazeLoss(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-4))

    # --- Load Datasets ---
    cfg['split'] = 'train'
    train_dataset = get_dataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    cfg['split'] = 'val'
    val_dataset = get_dataset(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Training and Validation Loop ---
    best_val_loss = float('inf')
    for epoch in range(cfg.get('epochs', 1)):
        # --- Training ---
        model.train()
        total_train_loss = 0
        for i, (imgs, binned_labels, cont_labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            binned_labels = binned_labels.to(device)
            cont_labels = cont_labels.to(device)

            optimizer.zero_grad()

            predictions = model(imgs)
            loss = criterion(predictions, binned_labels, cont_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (i + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{cfg['epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, binned_labels, cont_labels in val_loader:
                imgs = imgs.to(device)
                binned_labels = binned_labels.to(device)
                cont_labels = cont_labels.to(device)

                predictions = model(imgs)
                loss = criterion(predictions, binned_labels, cont_labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        logger.info("-------------------------")

        # --- Save Checkpoints ---
        # Save the latest model
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest.pth'))

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    logger.info("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file for training")
    args = parser.parse_args()
    main(args.config)
