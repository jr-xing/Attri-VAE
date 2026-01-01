#!/usr/bin/env python3
"""
Pre-train Deterministic Autoencoder for debugging VAE reconstruction issues.

This script trains a deterministic autoencoder (no KL divergence, no sampling)
to isolate whether blurry VAE reconstructions are due to:
- Architecture/data issues (AE also blurry)
- VAE-specific issues (AE is sharp)

Loss = MSE reconstruction only

Usage:
    python pretrain_ae.py --config configs/pretrain_ae.yaml

Author: Claude
Date: 2025-12-30
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
V1_STUDY_DIR = STUDY_DIR.parent / "SD_attrivae_age_ici"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(V1_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(V1_STUDY_DIR))

# Import model
from model.autoencoder import ConvAE, initialize_weights

# Import dataset from v1 study
from dataset import (
    AttriVAEDataset,
    create_patient_extractor,
    create_stratified_split,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten nested config for easier access
    flat_config = {}
    for section in ['data', 'model', 'training', 'loss']:
        if section in config:
            flat_config.update(config[section])

    # Add top-level keys
    for key in config:
        if key not in ['data', 'model', 'training', 'loss']:
            flat_config[key] = config[key]

    return flat_config


def save_reconstruction_visualization(
    model: nn.Module,
    val_loader: DataLoader,
    output_dir: Path,
    epoch: int,
    n_samples: int = 4,
):
    """Save visualization of input vs reconstruction.

    Args:
        model: Trained autoencoder
        val_loader: Validation data loader
        output_dir: Directory to save images
        epoch: Current epoch number
        n_samples: Number of samples to visualize
    """
    model.eval()

    # Get a batch
    data, label, attributes, _ = next(iter(val_loader))
    data = data[:n_samples].to(device)

    with torch.no_grad():
        recon, _ = model(data)

    # Convert to numpy
    data_np = data.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # Create figure with input/recon pairs
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        # Get middle slices
        mid_d = data_np.shape[2] // 2
        mid_h = data_np.shape[3] // 2
        mid_w = data_np.shape[4] // 2

        # Input - axial (middle depth slice)
        axes[i, 0].imshow(data_np[i, 0, mid_d, :, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input (axial)')
        axes[i, 0].axis('off')

        # Recon - axial
        axes[i, 1].imshow(recon_np[i, 0, mid_d, :, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Recon (axial)')
        axes[i, 1].axis('off')

        # Input - coronal (middle height slice)
        axes[i, 2].imshow(data_np[i, 0, :, mid_h, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Input (coronal)')
        axes[i, 2].axis('off')

        # Recon - coronal
        axes[i, 3].imshow(recon_np[i, 0, :, mid_h, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Recon (coronal)')
        axes[i, 3].axis('off')

    plt.suptitle(f'Epoch {epoch + 1}: Input vs Reconstruction', fontsize=14)
    plt.tight_layout()

    # Save figure
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    plt.savefig(vis_dir / f'recon_epoch_{epoch + 1:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch (reconstruction only).

    Args:
        epoch: Current epoch number
        model: ConvAE model
        train_loader: Training data loader
        optimizer: Optimizer
        config: Configuration dict

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0
    n_batches = 0

    for batch_idx, (data, label, attributes, _) in enumerate(train_loader):
        n_batches += 1

        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, z = model(data)

        # Reconstruction loss (MSE with mean reduction)
        loss = F.mse_loss(recon_batch, data, reduction='mean')

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_loss / n_batches,
    }

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: ConvAE model
        val_loader: Validation data loader

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (data, label, attributes, _) in enumerate(val_loader):
            n_batches += 1

            data = data.to(device)

            # Forward pass
            recon_batch, z = model(data)

            # Reconstruction loss (MSE)
            loss = F.mse_loss(recon_batch, data, reduction='mean')

            # Accumulate
            total_loss += loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_loss / n_batches,
    }

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path,
    filename: str = "checkpoint.pth",
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, output_dir / filename)


def main(args):
    """Main training function."""
    # Check if resuming from checkpoint
    checkpoint_data = None
    start_epoch = 0
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            checkpoint_file = resume_path / 'checkpoint.pth'
            if not checkpoint_file.exists():
                checkpoint_file = resume_path / 'best_model.pth'
            output_dir = resume_path
        else:
            checkpoint_file = resume_path
            output_dir = resume_path.parent

        print(f"[Resume] Loading checkpoint from {checkpoint_file}")
        checkpoint_data = torch.load(checkpoint_file, map_location=device, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        best_val_loss = checkpoint_data['metrics'].get('loss', float('inf'))

        # Load history if exists
        history_path = output_dir / 'history.npy'
        if history_path.exists():
            history = np.load(history_path, allow_pickle=True).item()
            print(f"[Resume] Loaded history with {len(history['train'])} epochs")

        # Load config from checkpoint directory
        config_path = output_dir / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
            print(f"[Resume] Loaded config from {config_path}")
        else:
            config = load_config(args.config)

        print(f"[Resume] Starting from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    else:
        # Load configuration
        config = load_config(args.config)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get('output_dir', STUDY_DIR / 'outputs' / 'pretrain_ae')) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    # Override with command line args
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size

    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config}")

    # Load data
    print("\n[Data] Loading dataset...")
    data = np.load(config['data_file'], allow_pickle=True).tolist()
    print(f"[Data] Loaded {len(data)} patients")

    # Create patient extractor for age/ICI
    extractor = create_patient_extractor(config['spreadsheet'])

    # Compute age statistics
    age_stats = compute_age_stats(data, extractor)
    print(f"[Data] Age stats: mean={age_stats['mean']:.1f}, std={age_stats['std']:.1f}")

    # Create train/val split
    train_data, val_data = create_stratified_split(
        data,
        val_fraction=config.get('val_fraction', 0.2),
        seed=config.get('seed', 42),
    )

    # Create datasets
    target_size = tuple(config.get('target_size', [80, 80, 80]))
    train_dataset = AttriVAEDataset(
        train_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=config.get('augment', True),
    )
    val_dataset = AttriVAEDataset(
        val_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )

    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # Create model
    print("\n[Model] Creating ConvAE (deterministic autoencoder)...")
    model = ConvAE(
        image_channels=config.get('image_channels', 1),
        h_dim=config.get('h_dim', 96),
        latent_size=config.get('latent_size', 64),
        n_filters_ENC=config.get('n_filters_enc', [8, 16, 32, 64, 2]),
        n_filters_DEC=config.get('n_filters_dec', [64, 32, 16, 8, 4, 2]),
    )
    model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    # Load checkpoint weights if resuming
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        print(f"[Resume] Loaded model and optimizer state from checkpoint")
    else:
        # Only initialize weights for fresh training
        model.apply(initialize_weights)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")

    # Training loop
    epochs = config.get('epochs', 200)

    if start_epoch > 0:
        print(f"\n[Training] Resuming AE pre-training from epoch {start_epoch + 1} to {epochs}...")
    else:
        print(f"\n[Training] Starting AE pre-training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, config)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(model, val_loader)
        history['val'].append(val_metrics)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train loss={train_metrics['loss']:.6f} | "
                f"Val loss={val_metrics['loss']:.6f}"
            )

        # Save visualization every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            save_reconstruction_visualization(model, val_loader, output_dir, epoch)

        # Save best model (based on validation loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename="best_model.pth"
            )

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename=f"checkpoint_epoch{epoch+1}.pth"
            )

        # Always save latest checkpoint (for resume)
        save_checkpoint(
            model, optimizer, epoch, val_metrics, config, output_dir,
            filename="checkpoint.pth"
        )

        # Save history after each epoch (for resume)
        np.save(output_dir / 'history.npy', history)

    # Save final model
    save_checkpoint(
        model, optimizer, epochs - 1, val_metrics, config, output_dir,
        filename="final_model.pth"
    )

    # Final visualization
    save_reconstruction_visualization(model, val_loader, output_dir, epochs - 1)

    # Save history
    np.save(output_dir / 'history.npy', history)

    # Plot training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    train_losses = [m['loss'] for m in history['train']]
    val_losses = [m['loss'] for m in history['val']]
    ax.plot(train_losses, label='Train MSE')
    ax.plot(val_losses, label='Val MSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Deterministic AE Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("=" * 70)
    print(f"[Done] Best validation loss: {best_val_loss:.6f}")
    print(f"[Done] Models saved to {output_dir}")
    print(f"\n[Diagnosis] Check visualizations in {output_dir}/visualizations/")
    print("  - If reconstructions are SHARP: VAE issue (KL weighting, posterior collapse)")
    print("  - If reconstructions are BLURRY: Architecture/data issue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train deterministic autoencoder")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(STUDY_DIR / 'configs' / 'pretrain_ae.yaml'),
        help='Path to config YAML'
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Resume from checkpoint (path to .pth file or output directory)'
    )

    args = parser.parse_args()
    main(args)
