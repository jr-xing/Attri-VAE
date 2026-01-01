#!/usr/bin/env python3
"""
Stage 1: Pre-train VAE for reconstruction only.

This script trains a VAE to learn good image reconstructions BEFORE adding
disentanglement losses. This two-stage approach ensures the model first
learns to reconstruct cardiac images well.

Loss = recon_loss + beta * kl_loss  (no MLP, no AR loss)

Usage:
    python pretrain_vae.py --config configs/pretrain.yaml

Author: Claude
Date: 2025-12-28
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
V1_STUDY_DIR = STUDY_DIR.parent / "SD_attrivae_age_ici"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(V1_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(V1_STUDY_DIR))

# Import model
from model.model import ConvVAE, initialize_weights

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


def ssim_3d(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Compute 3D Structural Similarity Index (SSIM).

    Args:
        img1: First image tensor (B, C, D, H, W)
        img2: Second image tensor (B, C, D, H, W)
        window_size: Size of the Gaussian window

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    # Create 3D Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 3D separable kernel
    window = g.view(1, 1, -1, 1, 1) * g.view(1, 1, 1, -1, 1) * g.view(1, 1, 1, 1, -1)
    window = window.expand(img1.size(1), 1, window_size, window_size, window_size)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute means
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv3d(img1 ** 2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv3d(img2 ** 2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM loss (1 - SSIM, so lower is better)."""
    return 1.0 - ssim_3d(img1, img2)


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


def get_beta(epoch: int, config: Dict[str, Any]) -> float:
    """Get KL weight with optional warmup.

    Args:
        epoch: Current epoch (0-indexed)
        config: Configuration dict

    Returns:
        Beta value for this epoch
    """
    warmup_epochs = config.get('beta_warmup_epochs', 50)
    beta_max = config.get('beta', 0.01)

    if epoch < warmup_epochs:
        # Linear warmup from 0 to beta_max
        return beta_max * (epoch / warmup_epochs)
    else:
        return beta_max


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch (reconstruction + KL only).

    Args:
        epoch: Current epoch number
        model: ConvVAE model
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
    total_kl = 0
    n_batches = 0

    # Get beta for this epoch (with warmup)
    beta = get_beta(epoch, config)
    ssim_weight = config.get('ssim_weight', 0.0)

    for batch_idx, (data, label, attributes, _) in enumerate(train_loader):
        n_batches += 1

        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)

        # Reconstruction loss (MSE with mean reduction)
        mse_loss = F.mse_loss(recon_batch, data, reduction='mean')

        # SSIM loss for sharper edges (optional)
        if ssim_weight > 0:
            ssim_l = ssim_loss(recon_batch, data)
            recon_loss = mse_loss + ssim_weight * ssim_l
        else:
            ssim_l = torch.tensor(0.0)
            recon_loss = mse_loss

        # KL divergence loss with FREE BITS to prevent posterior collapse
        # Free bits: only penalize KL if it's below a threshold per dimension
        # This ensures the latent space is used (prevents collapse to prior)
        kl_per_dim = torch.distributions.kl.kl_divergence(z_dist, prior_dist)  # (batch, latent_dim)
        free_bits = config.get('free_bits', 0.0)  # Minimum KL per dimension (e.g., 0.5)
        if free_bits > 0:
            # Apply free bits: max(kl, free_bits) - free_bits for each dim
            # This means KL below free_bits contributes 0 to loss
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = kl_per_dim.sum(1).mean()

        # Total loss = recon + beta * KL
        loss = config['recon_param'] * recon_loss + beta * kl_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_recon += mse_loss.item()
        total_ssim += ssim_l.item() if ssim_weight > 0 else 0
        total_kl += kl_loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'ssim_loss': total_ssim / n_batches,
        'kl_loss': total_kl / n_batches,
        'beta': beta,
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
        model: ConvVAE model
        val_loader: Validation data loader
        config: Configuration dict

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0
    total_recon = 0
    total_ssim = 0
    total_kl = 0
    n_batches = 0

    beta = config.get('beta', 0.01)  # Use full beta for validation
    ssim_weight = config.get('ssim_weight', 0.0)

    with torch.no_grad():
        for batch_idx, (data, label, attributes, _) in enumerate(val_loader):
            n_batches += 1

            data = data.to(device)

            # Forward pass
            recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)

            # Reconstruction loss (MSE)
            mse_loss = F.mse_loss(recon_batch, data, reduction='mean')

            # SSIM loss
            if ssim_weight > 0:
                ssim_l = ssim_loss(recon_batch, data)
                recon_loss = mse_loss + ssim_weight * ssim_l
            else:
                ssim_l = torch.tensor(0.0)
                recon_loss = mse_loss

            # KL divergence loss
            kl_loss = torch.distributions.kl.kl_divergence(z_dist, prior_dist).sum(1).mean()

            # Total loss
            loss = config['recon_param'] * recon_loss + beta * kl_loss

            # Accumulate
            total_loss += loss.item()
            total_recon += mse_loss.item()
            total_ssim += ssim_l.item() if ssim_weight > 0 else 0
            total_kl += kl_loss.item()

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'ssim_loss': total_ssim / n_batches,
        'kl_loss': total_kl / n_batches,
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
            # If directory given, look for checkpoint.pth or latest checkpoint
            checkpoint_file = resume_path / 'checkpoint.pth'
            if not checkpoint_file.exists():
                # Find latest epoch checkpoint
                checkpoints = list(resume_path.glob('checkpoint_epoch*.pth'))
                if checkpoints:
                    checkpoint_file = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
                else:
                    checkpoint_file = resume_path / 'best_recon_model.pth'
            output_dir = resume_path
        else:
            # If file given, use its parent as output dir
            checkpoint_file = resume_path
            output_dir = resume_path.parent

        print(f"[Resume] Loading checkpoint from {checkpoint_file}")
        checkpoint_data = torch.load(checkpoint_file, map_location=device, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        best_val_recon = checkpoint_data['metrics'].get('recon_loss', float('inf'))

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

        print(f"[Resume] Starting from epoch {start_epoch}, best_val_recon={best_val_recon:.6f}")
    else:
        # Load configuration
        config = load_config(args.config)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get('output_dir', STUDY_DIR / 'outputs' / 'pretrain')) / timestamp
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
    print("\n[Model] Creating ConvVAE...")
    model = ConvVAE(
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
    epochs = config.get('epochs', 300)

    if start_epoch > 0:
        print(f"\n[Training] Resuming pre-training from epoch {start_epoch + 1} to {epochs}...")
    else:
        print(f"\n[Training] Starting pre-training for {epochs} epochs...")
    print("=" * 70)

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
                f"Train: loss={train_metrics['loss']:.4f}, recon={train_metrics['recon_loss']:.6f}{ssim_str}, kl={train_metrics['kl_loss']:.4f}, beta={train_metrics['beta']:.4f} | "
                f"Val: recon={val_metrics['recon_loss']:.6f}"
            )

        # Save best model (based on validation reconstruction loss)
        if val_metrics['recon_loss'] < best_val_recon:
            best_val_recon = val_metrics['recon_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename="best_recon_model.pth"
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

    # Save history
    np.save(output_dir / 'history.npy', history)

    print("=" * 70)
    print(f"[Done] Best validation recon loss: {best_val_recon:.6f}")
    print(f"[Done] Models saved to {output_dir}")
    print(f"\nNext step: Fine-tune with disentanglement using:")
    print(f"  python finetune_attrivae.py --pretrained {output_dir}/best_recon_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train VAE for reconstruction")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(STUDY_DIR / 'configs' / 'pretrain.yaml'),
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
