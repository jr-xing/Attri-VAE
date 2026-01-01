#!/usr/bin/env python3
"""
Stage 2: Fine-tune pre-trained VAE with disentanglement losses.

This script loads a pre-trained VAE (from Stage 1) and adds:
- MLP classification loss (for CHIP prediction)
- Attribute regularization loss (for age/ICI disentanglement)

Loss = recon_loss + beta * kl_loss + alpha * mlp_loss + gamma * ar_loss

Usage:
    python finetune_attrivae.py --pretrained outputs/pretrain/<timestamp>/best_recon_model.pth

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
from sklearn.metrics import roc_auc_score
import yaml

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
V1_STUDY_DIR = STUDY_DIR.parent / "SD_attrivae_age_ici"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(V1_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(V1_STUDY_DIR))

# Import model and loss functions
from model.model import ConvVAE, initialize_weights
from model.loss_functions import mlp_loss_function, reg_loss

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


def load_pretrained_model(checkpoint_path: str, config: Dict[str, Any]) -> ConvVAE:
    """Load pre-trained VAE model.

    Args:
        checkpoint_path: Path to pre-trained checkpoint
        config: Model configuration

    Returns:
        Loaded model
    """
    model = ConvVAE(
        image_channels=config.get('image_channels', 1),
        h_dim=config.get('h_dim', 96),
        latent_size=config.get('latent_size', 64),
        n_filters_ENC=config.get('n_filters_enc', [8, 16, 32, 64, 2]),
        n_filters_DEC=config.get('n_filters_dec', [64, 32, 16, 8, 4, 2]),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"[Model] Loaded pre-trained model from {checkpoint_path}")

    if 'epoch' in checkpoint:
        print(f"[Model] Pre-trained for {checkpoint['epoch'] + 1} epochs")
    if 'metrics' in checkpoint:
        print(f"[Model] Pre-trained recon loss: {checkpoint['metrics'].get('recon_loss', 'N/A')}")

    return model


def safe_mean_accuracy(targets, weights):
    """Compute accuracy and AUC safely (handles single-class batches)."""
    weights = weights.squeeze()
    predictions = torch.zeros_like(weights)
    predictions[weights >= 0.5] = 1
    binary_targets = targets.float()
    correct = predictions == binary_targets
    acc = torch.sum(correct.float()) / binary_targets.view(-1).size(0)

    try:
        targets_np = binary_targets.detach().cpu().numpy()
        preds_np = weights.detach().cpu().numpy()
        if len(np.unique(targets_np)) > 1:
            score_roc_auc = roc_auc_score(targets_np, preds_np)
        else:
            score_roc_auc = 0.5
    except Exception:
        score_roc_auc = 0.5

    return acc, score_roc_auc


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch with all losses.

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
    total_kl = 0
    total_mlp = 0
    total_ar = 0
    total_acc = 0
    total_auc = 0
    n_batches = 0

    for batch_idx, (data, label, attributes, _) in enumerate(train_loader):
        n_batches += 1

        data = data.to(device)
        label = label.to(device)
        attributes = attributes.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)

        # Reconstruction loss (MSE with mean reduction)
        recon_loss = F.mse_loss(recon_batch, data, reduction='mean')

        # KL divergence loss
        kl_loss = torch.distributions.kl.kl_divergence(z_dist, prior_dist).sum(1).mean()

        # MLP classification loss
        mlp_loss = mlp_loss_function(label, out_mlp, config.get('alpha', 1.0))

        # Attribute regularization loss
        ar_loss = reg_loss(
            z_tilde, attributes, len(data),
            gamma=config.get('gamma', 1.0),
            factor=config.get('factor', 100.0)
        )

        # Total loss
        loss = (
            config.get('recon_param', 1.0) * recon_loss +
            config.get('beta', 0.01) * kl_loss +
            mlp_loss +
            ar_loss
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc, auc = safe_mean_accuracy(label, out_mlp)

        # Accumulate
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_mlp += mlp_loss.item()
        total_ar += ar_loss.item()
        total_acc += acc.item()
        total_auc += auc

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
        'mlp_loss': total_mlp / n_batches,
        'ar_loss': total_ar / n_batches,
        'accuracy': total_acc / n_batches,
        'auc': total_auc / n_batches,
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
    total_kl = 0
    total_mlp = 0
    total_ar = 0
    total_acc = 0
    total_auc = 0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (data, label, attributes, _) in enumerate(val_loader):
            n_batches += 1

            data = data.to(device)
            label = label.to(device)
            attributes = attributes.to(device)

            # Forward pass
            recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)

            # Losses
            recon_loss = F.mse_loss(recon_batch, data, reduction='mean')
            kl_loss = torch.distributions.kl.kl_divergence(z_dist, prior_dist).sum(1).mean()
            mlp_loss = mlp_loss_function(label, out_mlp, config.get('alpha', 1.0))
            ar_loss = reg_loss(
                z_tilde, attributes, len(data),
                gamma=config.get('gamma', 1.0),
                factor=config.get('factor', 100.0)
            )

            loss = (
                config.get('recon_param', 1.0) * recon_loss +
                config.get('beta', 0.01) * kl_loss +
                mlp_loss +
                ar_loss
            )

            # Compute accuracy
            acc, auc = safe_mean_accuracy(label, out_mlp)

            # Accumulate
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_mlp += mlp_loss.item()
            total_ar += ar_loss.item()
            total_acc += acc.item()
            total_auc += auc

    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
        'mlp_loss': total_mlp / n_batches,
        'ar_loss': total_ar / n_batches,
        'accuracy': total_acc / n_batches,
        'auc': total_auc / n_batches,
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
    """Main fine-tuning function."""
    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get('output_dir', STUDY_DIR / 'outputs' / 'finetune')) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config['pretrained_model'] = args.pretrained
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

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

    # Load pre-trained model
    print("\n[Model] Loading pre-trained VAE...")
    model = load_pretrained_model(args.pretrained, config)
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")

    # Create optimizer with lower learning rate for fine-tuning
    lr = config.get('learning_rate', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"[Optimizer] Learning rate: {lr}")

    # Training loop
    epochs = config.get('epochs', 300)
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_val_recon = float('inf')
    history = {'train': [], 'val': []}

    print(f"\n[Training] Starting fine-tuning for {epochs} epochs...")
    print("=" * 100)

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, config)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(epoch, model, val_loader, config)
        history['val'].append(val_metrics)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: loss={train_metrics['loss']:.4f}, recon={train_metrics['recon_loss']:.6f}, ar={train_metrics['ar_loss']:.4f}, acc={train_metrics['accuracy']:.3f} | "
                f"Val: recon={val_metrics['recon_loss']:.6f}, acc={val_metrics['accuracy']:.3f}, auc={val_metrics['auc']:.3f}"
            )

        # Save best models
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename="best_loss_model.pth"
            )

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config, output_dir,
                filename="best_auc_model.pth"
            )

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

    # Save final model
    save_checkpoint(
        model, optimizer, epochs - 1, val_metrics, config, output_dir,
        filename="final_model.pth"
    )

    # Save history
    np.save(output_dir / 'history.npy', history)

    print("=" * 100)
    print(f"[Done] Best validation loss: {best_val_loss:.4f}")
    print(f"[Done] Best validation AUC: {best_val_auc:.3f}")
    print(f"[Done] Best validation recon: {best_val_recon:.6f}")
    print(f"[Done] Models saved to {output_dir}")
    print(f"\nVisualize results with:")
    print(f"  python visualize.py --checkpoint {output_dir}/best_auc_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VAE with disentanglement")
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=True,
        help='Path to pre-trained VAE checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(STUDY_DIR / 'configs' / 'finetune.yaml'),
        help='Path to config YAML'
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')

    args = parser.parse_args()
    main(args)
