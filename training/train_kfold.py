import argparse
import os
import random
import time
from typing import Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.datasets import get_dataset
from src.models import build_model
from src.models.backbone_resnet import ResNetBackbone
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.losses import GazeLoss


@runtime_checkable
class HasFreezeBNStats(Protocol):
    def freeze_bn_stats(self) -> None: ...


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model, cfg, logger):
    """Creates optimizer with proper parameter handling for different backbones."""
    if isinstance(model.backbone, ResNetBackbone):
        logger.info(
            "Using ResNet fine-tuning strategy (freezing conv1 and bn1 layers)."
        )
        for param in model.backbone.net.conv1.parameters():
            param.requires_grad = False
        for param in model.backbone.net.bn1.parameters():
            param.requires_grad = False
        fine_tune_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            fine_tune_params, lr=cfg.get("lr", 1e-4), weight_decay=1e-4
        )
    else:
        logger.info("Using standard training strategy (all layers trainable).")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=1e-4
        )
    return optimizer


def train_one_epoch(
    model: torch.nn.Module,
    criterion: GazeLoss,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    logger,
    scheduler=None,
    log_interval=100,
):
    """Performs one full training epoch."""
    model.train()

    if isinstance(model.backbone, ResNetBackbone) and isinstance(
        model.backbone, HasFreezeBNStats
    ):
        model.backbone.freeze_bn_stats()

    total_train_loss = 0
    for i, (imgs, binned_labels, cont_labels) in enumerate(data_loader):
        imgs, binned_labels, cont_labels = (
            imgs.to(device),
            binned_labels.to(device),
            cont_labels.to(device),
        )
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, binned_labels, cont_labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_train_loss += loss.item()

        if (i + 1) % log_interval == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{total_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}"
            )

    return total_train_loss / len(data_loader)


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    criterion: GazeLoss,
    data_loader: DataLoader,
    device: torch.device,
):
    """Performs one full validation epoch."""
    model.eval()
    total_val_loss = 0
    for imgs, binned_labels, cont_labels in data_loader:
        imgs, binned_labels, cont_labels = (
            imgs.to(device),
            binned_labels.to(device),
            cont_labels.to(device),
        )
        predictions = model(imgs)
        loss = criterion(predictions, binned_labels, cont_labels)
        total_val_loss += loss.item()
    return total_val_loss / len(data_loader)


def train_single_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    fold_output_dir: str,
    logger,
):
    """Trains a single fold and returns the best validation loss and epoch."""
    logger.info(f"=== Starting Fold {fold + 1} ===")

    # Create model, criterion, optimizer, scheduler for this fold
    model = build_model(cfg).to(device)
    criterion = GazeLoss(cfg).to(device)
    optimizer = create_optimizer(model, cfg, logger)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.get("lr", 1e-4) * 20,
        epochs=cfg["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        anneal_strategy="cos",
    )

    # Training loop for this fold
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(cfg["epochs"]):
        # Train
        avg_train_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            cfg["epochs"],
            logger,
            scheduler=scheduler,
            log_interval=200,  # Less frequent logging for cross-validation
        )
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss = evaluate_one_epoch(model, criterion, val_loader, device)
        val_losses.append(avg_val_loss)

        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(), os.path.join(fold_output_dir, "best_fold.pth")
            )

        # Log every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{cfg['epochs']}: "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

    # Save final model for this fold
    torch.save(model.state_dict(), os.path.join(fold_output_dir, "final_fold.pth"))

    # Save fold results
    fold_results = {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    # Plot fold loss curve
    plot_fold_loss_curve(train_losses, val_losses, fold, fold_output_dir)

    logger.info(
        f"=== Fold {fold + 1} Complete ===\n"
        f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})\n"
        f"Final Val Loss: {val_losses[-1]:.4f}"
    )

    return fold_results


def plot_fold_loss_curve(train_losses, val_losses, fold, output_dir):
    """Generates and saves a plot for a single fold's loss curves."""
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title(f"Fold {fold + 1} - Training and Validation Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    best_epoch = val_losses.index(min(val_losses))
    best_loss = min(val_losses)
    plt.scatter(
        best_epoch,
        best_loss,
        color="red",
        zorder=5,
        label=f"Best Val Loss: {best_loss:.4f}",
    )
    plt.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1)

    plot_path = os.path.join(output_dir, f"fold_{fold + 1}_loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_cv_summary(cv_results, output_dir):
    """Creates summary plots for cross-validation results."""
    # Plot 1: Best validation loss per fold
    plt.figure(figsize=(12, 8))
    folds = [r["fold"] + 1 for r in cv_results]
    best_losses = [r["best_val_loss"] for r in cv_results]
    final_losses = [r["final_val_loss"] for r in cv_results]

    x = np.arange(len(folds))
    width = 0.35

    plt.bar(x - width / 2, best_losses, width, label="Best Val Loss", alpha=0.8)
    plt.bar(x + width / 2, final_losses, width, label="Final Val Loss", alpha=0.8)

    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Cross-Validation Results: Best vs Final Validation Loss")
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add mean lines
    mean_best = np.mean(best_losses)
    mean_final = np.mean(final_losses)
    plt.axhline(
        y=mean_best,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Best: {mean_best:.4f}",
    )
    plt.axhline(
        y=mean_final,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Final: {mean_final:.4f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_summary.png"))
    plt.close()

    # Plot 2: All fold loss curves together
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(cv_results):
        plt.plot(result["val_losses"], label=f"Fold {i + 1}", alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("All Folds Validation Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "all_folds_curves.png"))
    plt.close()


def train_final_model_on_all_data(
    combined_dataset, best_epoch, cfg, device, output_dir, logger
):
    """Trains the final production model on all available data for the optimal number of epochs."""
    logger.info("=== Training Final Production Model on All Data ===")

    # Create data loader for all data
    final_loader = DataLoader(
        combined_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create model, criterion, optimizer, scheduler
    model = build_model(cfg).to(device)
    criterion = GazeLoss(cfg).to(device)
    optimizer = create_optimizer(model, cfg, logger)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.get("lr", 1e-4) * 20,
        epochs=best_epoch + 1,  # Train for optimal number of epochs
        steps_per_epoch=len(final_loader),
        pct_start=0.25,
        anneal_strategy="cos",
    )

    logger.info(
        f"Training final model for {best_epoch + 1} epochs on {len(combined_dataset)} samples"
    )

    # Training loop
    final_train_losses = []
    for epoch in range(best_epoch + 1):
        avg_train_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            final_loader,
            device,
            epoch,
            best_epoch + 1,
            logger,
            scheduler=scheduler,
            log_interval=100,
        )
        final_train_losses.append(avg_train_loss)

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Final Model - Epoch {epoch + 1}/{best_epoch + 1}: Train Loss: {avg_train_loss:.4f}"
            )

    # Save final production model
    final_model_path = os.path.join(output_dir, "final_production_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Plot final model training curve
    plt.figure(figsize=(12, 7))
    plt.plot(
        final_train_losses,
        label="Final Model Training Loss",
        color="green",
        linewidth=2,
    )
    plt.title("Final Production Model Training Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "final_model_training_curve.png"))
    plt.close()

    logger.info("=== Final Production Model Saved ===")
    logger.info(f"Model path: {final_model_path}")
    logger.info(
        f"Trained on {len(combined_dataset)} samples for {best_epoch + 1} epochs"
    )
    logger.info(f"Final training loss: {final_train_losses[-1]:.4f}")

    return final_model_path, final_train_losses


def main(cfg_path, k_folds=5, seed=42):
    """Main K-Fold Cross-Validation training orchestrator."""
    cfg = load_config(cfg_path)

    # Override K-fold specific parameters
    cfg["k_folds"] = k_folds
    cfg["seed"] = seed

    # Setup
    set_random_seeds(seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "output", f"kfold_{cfg['dataset_name']}_{cfg['backbone']}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(os.path.join(output_dir, "kfold_train.log"))

    # Log configuration
    logger.info("--- K-Fold Cross-Validation Training Configuration ---")
    logger.info(f"K-Folds: {k_folds}")
    logger.info(f"Random Seed: {seed}")
    logger.info("Original Config:\n%s", yaml.dump(cfg, default_flow_style=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and combine all data (train + val + test)
    # We'll override the split parameter for each dataset
    train_dataset = get_dataset({**cfg, "split": "train"})
    val_dataset = get_dataset({**cfg, "split": "val"})
    test_dataset = get_dataset({**cfg, "split": "test"})

    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    logger.info(f"Total dataset size: {len(combined_dataset)} samples")
    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # K-Fold Cross-Validation setup
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    logger.info(f"Starting {k_folds}-Fold Cross-Validation...")

    # Store results from all folds
    cv_results = []

    # K-Fold training loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(combined_dataset)):
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Create train and validation subsets for this fold
        train_subset = Subset(combined_dataset, train_idx)
        val_subset = Subset(combined_dataset, val_idx)

        logger.info(
            f"Fold {fold + 1}: Train={len(train_subset)}, Val={len(val_subset)}"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Train this fold
        fold_result = train_single_fold(
            fold, train_loader, val_loader, cfg, device, fold_output_dir, logger
        )
        cv_results.append(fold_result)

    # Analyze cross-validation results
    best_val_losses = [r["best_val_loss"] for r in cv_results]
    best_epochs = [r["best_epoch"] for r in cv_results]

    mean_best_loss = np.mean(best_val_losses)
    std_best_loss = np.std(best_val_losses)
    mean_best_epoch = int(np.mean(best_epochs))

    logger.info("=== Cross-Validation Results Summary ===")
    logger.info(
        f"Mean Best Validation Loss: {mean_best_loss:.4f} ± {std_best_loss:.4f}"
    )
    logger.info(
        f"Best Validation Losses: {[f'{loss:.4f}' for loss in best_val_losses]}"
    )
    logger.info(f"Best Epochs: {best_epochs}")
    logger.info(f"Mean Best Epoch: {mean_best_epoch}")

    # Save cross-validation results
    cv_summary = {
        "mean_best_val_loss": mean_best_loss,
        "std_best_val_loss": std_best_loss,
        "mean_best_epoch": mean_best_epoch,
        "best_val_losses": best_val_losses,
        "best_epochs": best_epochs,
        "fold_results": cv_results,
        "config": cfg,
    }

    import json

    with open(os.path.join(output_dir, "cv_results.json"), "w") as f:
        json.dump(
            cv_summary,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )

    # Create summary plots
    plot_cv_summary(cv_results, output_dir)

    # Train final production model on all data
    final_model_path, final_train_losses = train_final_model_on_all_data(
        combined_dataset, mean_best_epoch, cfg, device, output_dir, logger
    )

    logger.info("=== K-Fold Cross-Validation Complete ===")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Final production model: {final_model_path}")
    logger.info(f"CV Performance: {mean_best_loss:.4f} ± {std_best_loss:.4f}")

    print(output_dir)  # END script by printing output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file for K-fold training",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    main(args.config, args.k_folds, args.seed)
