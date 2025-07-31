import yaml
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
import time
import matplotlib.pyplot as plt

from src.models import build_model
from src.datasets import get_dataset
from src.utils.losses import GazeLoss
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.models.backbone_resnet import ResNetBackbone

from typing import Protocol, runtime_checkable

@runtime_checkable
class HasFreezeBNStats(Protocol):
    def freeze_bn_stats(self) -> None:
        ...

# --- Helper Function for a Single Training Epoch ---
def train_one_epoch(model: torch.nn.Module,
                    criterion: GazeLoss,
                    optimizer: Optimizer,
                    data_loader: DataLoader,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    logger):
    """
    Performs one full training epoch.
    """
    model.train()
    if isinstance(model.backbone, ResNetBackbone) and isinstance(model.backbone, HasFreezeBNStats):
        model.backbone.freeze_bn_stats()

    total_train_loss = 0
    for i, (imgs, binned_labels, cont_labels) in enumerate(data_loader):
        imgs, binned_labels, cont_labels = imgs.to(device), binned_labels.to(device), cont_labels.to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, binned_labels, cont_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        if (i + 1) % 100 == 0:
            logger.info(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    return total_train_loss / len(data_loader)

# --- Helper Function for a Single Validation Epoch ---
@torch.no_grad()
def evaluate_one_epoch(model: torch.nn.Module,
                       criterion: GazeLoss,
                       data_loader: DataLoader,
                       device: torch.device):
    """
    Performs one full validation epoch.
    """
    model.eval()
    total_val_loss = 0
    for imgs, binned_labels, cont_labels in data_loader:
        imgs, binned_labels, cont_labels = imgs.to(device), binned_labels.to(device), cont_labels.to(device)
        predictions = model(imgs)
        loss = criterion(predictions, binned_labels, cont_labels)
        total_val_loss += loss.item()
    return total_val_loss / len(data_loader)


def plot_loss_curve(train_losses, val_losses, output_dir):
    """Generates and saves a plot of training and validation loss curves."""
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if val_losses:
        best_epoch = val_losses.index(min(val_losses))
        best_loss = min(val_losses)
        plt.scatter(best_epoch, best_loss, color='red', zorder=5, label=f'Best Val Loss: {best_loss:.4f}')
        plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1)
    plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main(cfg_path):
    """Main training orchestrator."""
    cfg = load_config(cfg_path)

    # Setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('output', f"{cfg['dataset_name']}_{cfg['backbone']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(os.path.join(output_dir, 'train.log'))
    logger.info("--- Training Configuration ---\n%s", yaml.dump(cfg))
    do_validation = cfg.get('do_validation', True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Model, Criterion, Optimizer, Scheduler
    model = build_model(cfg).to(device)
    criterion = GazeLoss(cfg).to(device)

    if isinstance(model.backbone, ResNetBackbone):
        logger.info("Using ResNet fine-tuning strategy (freezing conv1 and bn1 layers).")
        for param in model.backbone.net.conv1.parameters(): param.requires_grad = False
        for param in model.backbone.net.bn1.parameters(): param.requires_grad = False
        fine_tune_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(fine_tune_params, lr=cfg.get('lr', 1e-4), weight_decay=1e-4)
    else:
        logger.info("Using standard training strategy (all layers trainable).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 1e-4), weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-7)

    # DataLoaders
    train_loader = DataLoader(get_dataset({**cfg, 'split': 'train'}), batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(get_dataset({**cfg, 'split': 'val'}), batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True) if do_validation else None

    # Training Loop
    best_val_loss = float('inf')
    train_loss_history, val_loss_history = [], []
    logger.info("Starting training...")
    for epoch in range(cfg['epochs']):
        avg_train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, cfg['epochs'], logger)
        train_loss_history.append(avg_train_loss)

        logger.info(f"--- Epoch {epoch+1}/{cfg['epochs']} Summary ---")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")

        if do_validation and val_loader:
            avg_val_loss = evaluate_one_epoch(model, criterion, val_loader, device)
            val_loss_history.append(avg_val_loss)
            logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{epoch + 1}.pth'))

        torch.save(model.state_dict(), os.path.join(output_dir, 'latest.pth'))
        scheduler.step()
        logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.8f}")
        logger.info("-------------------------")

    # Finalize
    logger.info("Training complete!")
    if do_validation:
        plot_path = plot_loss_curve(train_loss_history, val_loss_history, output_dir)
        logger.info(f"Loss curve plot saved to {plot_path}")

    print(output_dir) # END script by printing output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file for training")
    args = parser.parse_args()
    main(args.config)
