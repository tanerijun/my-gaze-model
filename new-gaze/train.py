import yaml
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import build_model
from src.datasets import get_dataset
from src.utils.losses import GazeLoss
from src.utils.logger import get_logger
import time
from src.models.backbone_resnet import ResNetBackbone
import matplotlib.pyplot as plt
from typing import Protocol, runtime_checkable

@runtime_checkable
class HasFreezeBNStats(Protocol):
    def freeze_bn_stats(self) -> None:
        ...

def plot_loss_curve(train_losses, val_losses, output_dir):
    """
    Generates and saves a plot of training and validation loss curves.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Find the epoch with the best validation loss for annotation
    best_epoch = val_losses.index(min(val_losses))
    best_loss = min(val_losses)
    plt.scatter(best_epoch, best_loss, color='red', zorder=5, label=f'Best Val Loss: {best_loss:.4f}')
    plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1)

    plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    plt.close() # Close the figure to free up memory
    return plot_path

def main(cfg_path):
    """
    Main training loop for gaze estimation.
    """
    # --- Load Configuration and Logger ---
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

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

    # --- Optimizer Setup ---
    backbone = model.backbone
    if isinstance(backbone, ResNetBackbone):
        logger.info("Using ResNet fine-tuning strategy (freezing conv1 and bn1 layers).")
        conv1_params = backbone.net.conv1.parameters()
        bn1_params = backbone.net.bn1.parameters()
        for param in conv1_params: param.requires_grad = False
        for param in bn1_params: param.requires_grad = False
        fine_tune_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(fine_tune_params, lr=cfg.get('lr', 1e-4))
    else:
        logger.info("Using standard training strategy (all layers trainable).")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-4))

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-7)

    # --- Load Datasets ---
    train_dataset = get_dataset({**cfg, 'split': 'train'})
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = get_dataset({**cfg, 'split': 'val'})
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # --- Lists to store loss history for plotting ---
    train_loss_history = []
    val_loss_history = []

    # --- Training and Validation Loop ---
    best_val_loss = float('inf')
    for epoch in range(cfg.get('epochs')):
        model.train()

        if isinstance(model.backbone, ResNetBackbone) and isinstance(model.backbone, HasFreezeBNStats):
            model.backbone.freeze_bn_stats()

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
        train_loss_history.append(avg_train_loss)
        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, binned_labels, cont_labels in val_loader:
                imgs, binned_labels, cont_labels = imgs.to(device), binned_labels.to(device), cont_labels.to(device)
                predictions = model(imgs)
                loss = criterion(predictions, binned_labels, cont_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        logger.info("-------------------------")

        scheduler.step()
        logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.8f}")

        torch.save(model.state_dict(), os.path.join(output_dir, 'latest.pth'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    logger.info("Training complete!")

    # --- Plotting Loss Curve ---
    plot_path = plot_loss_curve(train_loss_history, val_loss_history, output_dir)
    logger.info(f"Loss curve plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file for training")
    args = parser.parse_args()
    main(args.config)
