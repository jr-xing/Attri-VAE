#!/usr/bin/env python3
"""
Pre-train 2D VQ-VAE for reconstruction.

VQ-VAE uses discrete latent codes (codebook) instead of continuous Gaussian latents.
This typically produces sharper reconstructions because:
- No averaging over continuous distributions
- Each spatial location maps to ONE discrete code
- Forces the model to commit to specific reconstructions

Loss = recon_loss + vq_loss (+ optional ssim_loss)

Key differences from standard VAE:
- No KL divergence (discrete latents, not Gaussian)
- No beta warmup needed
- VQ loss provides regularization naturally

Usage:
    python pretrain_vqvae_2d.py --config configs/pretrain_vqvae.yaml

Author: Claude
Date: 2025-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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
V3_DIR = PROJECT_ROOT / 'studies' / 'SD_attrivae_age_ici_v3_2D'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

# Import VQ-VAE model
from model_vqvae_2d import VQVAE2D, get_model

# Import dataset from v3
from dataset_2d import (
    AttriVAEDataset2D,
    create_patient_extractor,
    create_stratified_split,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def ssim_2d(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Compute 2D Structural Similarity Index (SSIM).

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        window_size: Size of the Gaussian window

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    # Create 2D Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 2D separable kernel
    window = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    window = window.expand(img1.size(1), 1, window_size, window_size)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM loss (1 - SSIM, so lower is better)."""
    return 1.0 - ssim_2d(img1, img2)


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


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch.

    VQ-VAE Loss = recon_loss + vq_loss (+ optional ssim_loss)

    Args:
        epoch: Current epoch number
        model: VQVAE2D model
        train_loader: Training data loader
        optimizer: Optimizer
        config: Configuration dict

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0
    total_recon = 0
    total_ssim = 0
    total_vq = 0
    n_batches = 0

    ssim_weight = config.get('ssim_weight', 0.0)

    for batch_idx, (data, label, attributes, subject_id, slice_idx) in enumerate(train_loader):
        n_batches += 1

        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, input_batch, vq_loss = model(data)

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(recon_batch, data, reduction='mean')

        # SSIM loss for sharper edges (optional)
        if ssim_weight > 0:
            ssim_l = ssim_loss(recon_batch, data)
            recon_loss = mse_loss + ssim_weight * ssim_l
        else:
            ssim_l = torch.tensor(0.0)
            recon_loss = mse_loss

        # Total loss = recon + vq_loss (no KL for VQ-VAE!)
        loss = recon_loss + vq_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_recon += mse_loss.item()
        total_ssim += ssim_l.item() if ssim_weight > 0 else 0
        total_vq += vq_loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'ssim_loss': total_ssim / n_batches,
        'vq_loss': total_vq / n_batches,
    }

    return metrics


def validate(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Validate the model.

    Args:
        epoch: Current epoch number
        model: VQVAE2D model
        val_loader: Validation data loader
        config: Configuration dict

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0
    total_recon = 0
    total_ssim = 0
    total_vq = 0
    n_batches = 0

    ssim_weight = config.get('ssim_weight', 0.0)

    with torch.no_grad():
        for batch_idx, (data, label, attributes, subject_id, slice_idx) in enumerate(val_loader):
            n_batches += 1

            data = data.to(device)

            # Forward pass
            recon_batch, input_batch, vq_loss = model(data)

            # Reconstruction loss (MSE)
            mse_loss = F.mse_loss(recon_batch, data, reduction='mean')

            # SSIM loss
            if ssim_weight > 0:
                ssim_l = ssim_loss(recon_batch, data)
                recon_loss = mse_loss + ssim_weight * ssim_l
            else:
                ssim_l = torch.tensor(0.0)
                recon_loss = mse_loss

            # Total loss
            loss = recon_loss + vq_loss

            # Accumulate
            total_loss += loss.item()
            total_recon += mse_loss.item()
            total_ssim += ssim_l.item() if ssim_weight > 0 else 0
            total_vq += vq_loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'ssim_loss': total_ssim / n_batches,
        'vq_loss': total_vq / n_batches,
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
    best_val_recon = float('inf')

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            checkpoint_file = resume_path / 'checkpoint.pth'
            if not checkpoint_file.exists():
                checkpoints = list(resume_path.glob('checkpoint_epoch*.pth'))
                if checkpoints:
                    checkpoint_file = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
                else:
                    checkpoint_file = resume_path / 'best_recon_model.pth'
            output_dir = resume_path
        else:
            checkpoint_file = resume_path
            output_dir = resume_path.parent

        print(f"[Resume] Loading checkpoint from {checkpoint_file}")
        checkpoint_data = torch.load(checkpoint_file, map_location=device, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        best_val_recon = checkpoint_data['metrics'].get('recon_loss', float('inf'))

        history_path = output_dir / 'history.npy'
        if history_path.exists():
            history = np.load(history_path, allow_pickle=True).item()
            print(f"[Resume] Loaded history with {len(history['train'])} epochs")

        config_path = output_dir / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
            print(f"[Resume] Loaded config from {config_path}")
        else:
            config = load_config(args.config)

        print(f"[Resume] Starting from epoch {start_epoch}, best_val_recon={best_val_recon:.6f}")
    else:
        # Load configuration
        config = load_config(args.config)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get('output_dir', STUDY_DIR / 'outputs' / 'pretrain_vqvae')) / timestamp
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

    # Create train/val split (at patient level)
    train_data, val_data = create_stratified_split(
        data,
        val_fraction=config.get('val_fraction', 0.2),
        seed=config.get('seed', 42),
    )

    # Create 2D datasets
    target_size = (config.get('img_size', 80), config.get('img_size', 80))
    train_dataset = AttriVAEDataset2D(
        train_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=config.get('augment', True),
    )
    val_dataset = AttriVAEDataset2D(
        val_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )

    print(f"[Data] Train: {len(train_dataset)} slices, Val: {len(val_dataset)} slices")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # Create VQ-VAE model
    print("\n[Model] Creating VQVAE2D...")
    model = VQVAE2D(
        in_channels=config.get('image_channels', 1),
        embedding_dim=config.get('embedding_dim', 64),
        num_embeddings=config.get('num_embeddings', 512),
        hidden_dims=config.get('hidden_dims', [128, 256]),
        beta=config.get('beta', 0.25),
        img_size=config.get('img_size', 80),
    )
    model.to(device)

    # Create optimizer with weight decay for regularization
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=weight_decay
    )
    if weight_decay > 0:
        print(f"[Model] Using weight decay: {weight_decay}")

    # Load checkpoint weights if resuming
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        print(f"[Resume] Loaded model and optimizer state from checkpoint")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")
    print(f"[Model] Codebook: {config.get('num_embeddings', 512)} codes x {config.get('embedding_dim', 64)} dim")

    # Training loop
    epochs = config.get('epochs', 300)

    # Early stopping configuration
    early_stopping = config.get('early_stopping', False)
    patience = config.get('patience', 50)
    min_delta = config.get('min_delta', 0.0001)
    epochs_without_improvement = 0
    best_epoch = 0

    if start_epoch > 0:
        print(f"\n[Training] Resuming VQ-VAE pre-training from epoch {start_epoch + 1} to {epochs}...")
    else:
        print(f"\n[Training] Starting VQ-VAE pre-training for {epochs} epochs...")
    if early_stopping:
        print(f"[Training] Early stopping enabled: patience={patience}, min_delta={min_delta}")
    print("=" * 80)

    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, config)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(epoch, model, val_loader, config)
        history['val'].append(val_metrics)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ssim_str = f", ssim={train_metrics['ssim_loss']:.4f}" if train_metrics.get('ssim_loss', 0) > 0 else ""
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: loss={train_metrics['loss']:.4f}, recon={train_metrics['recon_loss']:.6f}{ssim_str}, vq={train_metrics['vq_loss']:.4f} | "
                f"Val: recon={val_metrics['recon_loss']:.6f}, vq={val_metrics['vq_loss']:.4f}"
            )

        # Save best model (based on validation reconstruction loss)
        if val_metrics['recon_loss'] < best_val_recon - min_delta:
            best_val_recon = val_metrics['recon_loss']
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename="best_recon_model.pth"
            )
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stopping and epochs_without_improvement >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch + 1}.")
            print(f"[Early Stopping] Best epoch was {best_epoch + 1} with val_recon={best_val_recon:.6f}")
            break

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

        # Save history after each epoch
        np.save(output_dir / 'history.npy', history)

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_metrics, config, output_dir,
        filename="final_model.pth"
    )

    # Save history
    np.save(output_dir / 'history.npy', history)

    print("=" * 80)
    print(f"[Done] Best validation recon loss: {best_val_recon:.6f}")
    print(f"[Done] Models saved to {output_dir}")
    print(f"\nTo visualize results:")
    print(f"  python visualize_vqvae.py --model {output_dir}/best_recon_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train VQ-VAE for reconstruction")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(STUDY_DIR / 'configs' / 'pretrain_vqvae.yaml'),
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
