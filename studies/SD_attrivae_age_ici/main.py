#!/usr/bin/env python3
"""
Main training script for AttriVAE with Age/ICI attributes.

This script trains an AttriVAE model to learn disentangled representations
where latent dimension 0 correlates with age and dimension 1 with ICI status.

Usage:
    python main.py [--config CONFIG_PATH] [--epochs N] [--resume]

Author: Claude
Date: 2025-12-28
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
LGE_CHIP_ROOT = Path("/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

# Import model and loss functions
from model.model import ConvVAE, initialize_weights
from model.loss_functions import (
    KL_loss,
    mlp_loss_function,
    reg_loss,
)
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def reconstruction_loss_mean(recon_x, x, recon_param):
    """Reconstruction loss using MEAN reduction (not sum).

    The original loss_functions.py uses sum reduction which gives very large
    values for 80x80x80 volumes (~512K voxels). Using mean reduction gives
    values in a reasonable range that can be balanced with other losses.
    """
    # MSE with mean reduction - gives per-voxel average error
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    return recon_param * recon_loss


def safe_mean_accuracy(targets, weights):
    """Compute mean accuracy with safe AUC computation.

    Handles cases where only one class is present in the batch.
    """
    weights = weights.squeeze()
    predictions = torch.zeros_like(weights)
    predictions[weights >= 0.5] = 1
    binary_targets = targets.float()
    correct = predictions == binary_targets
    acc = torch.sum(correct.float()) / binary_targets.view(-1).size(0)

    # Safe AUC computation using probabilities
    try:
        targets_np = binary_targets.detach().cpu().numpy()
        preds_np = weights.detach().cpu().numpy()  # Use probabilities
        if len(np.unique(targets_np)) > 1:
            score_roc_auc = roc_auc_score(targets_np, preds_np)
        else:
            score_roc_auc = 0.5  # Undefined, return 0.5
    except Exception:
        score_roc_auc = 0.5

    return acc, score_roc_auc

# Import dataset
from dataset import (
    AttriVAEDataset,
    create_patient_extractor,
    create_stratified_split,
    compute_age_stats
)

# Check CUDA
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Using device: {device}")

# Default paths
DEFAULT_DATA_FILE = LGE_CHIP_ROOT / "data" / "2025-06-01-onc-cohort-144-with-serial-scans-and-103-LGE-masks.npy"
DEFAULT_SPREADSHEET = LGE_CHIP_ROOT / "link_project_data" / "CHIP ICI MI CM outcomes - Updated 2025-10-09.xlsx"


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        # Data
        'data_file': str(DEFAULT_DATA_FILE),
        'spreadsheet': str(DEFAULT_SPREADSHEET),
        'val_fraction': 0.2,
        'seed': 42,

        # Model
        'image_channels': 1,
        'h_dim': 96,
        'latent_size': 64,
        'n_filters_enc': (8, 16, 32, 64, 2),
        'n_filters_dec': (64, 32, 16, 8, 4, 2),
        'target_size': (80, 80, 80),

        # Training
        'batch_size': 8,
        'epochs': 500,
        'learning_rate': 0.0001,
        'num_workers': 4,

        # Loss weights
        'recon_param': 1.0,
        'beta': 2.0,      # KL weight
        'alpha': 1.0,     # Classification (MLP) weight
        'gamma': 10.0,    # Attribute regularization weight
        'factor': 100.0,  # tanh scaling in AR loss

        # Flags
        'use_AR_LOSS': True,
        'is_L1': False,
        'augment': True,

        # Output
        'output_dir': str(STUDY_DIR / "outputs" / "training"),
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    config = get_default_config()

    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            # Flatten nested config if present
            if 'data' in yaml_config:
                config.update(yaml_config['data'])
            if 'model' in yaml_config:
                config.update(yaml_config['model'])
            if 'training' in yaml_config:
                config.update(yaml_config['training'])
            if 'loss' in yaml_config:
                config.update(yaml_config['loss'])
            # Top-level overrides
            for key in ['use_AR_LOSS', 'is_L1', 'output_dir']:
                if key in yaml_config:
                    config[key] = yaml_config[key]
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")

    return config


def save_checkpoint(state: Dict, checkpoint_dir: Path, filename: str = 'checkpoint.pth'):
    """Save checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train for one epoch.

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

        # Compute losses - using MEAN reduction for reconstruction
        recon_loss = reconstruction_loss_mean(recon_batch, data, config['recon_param'])
        mlp_loss = mlp_loss_function(label, out_mlp, config['alpha'])

        # KL annealing: gradually increase KL weight over warmup epochs
        warmup_epochs = config.get('kl_warmup_epochs', 50)
        if epoch < warmup_epochs:
            kl_weight = config['beta'] * (epoch / warmup_epochs)
        else:
            kl_weight = config['beta']

        kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, kl_weight, c=0.0)

        loss = recon_loss + mlp_loss + kl_loss2

        # Attribute regularization loss with warmup
        ar_loss = torch.tensor(0.0, device=device)
        if config['use_AR_LOSS']:
            # Also warm up AR loss
            if epoch < warmup_epochs:
                ar_weight = config['gamma'] * (epoch / warmup_epochs)
            else:
                ar_weight = config['gamma']
            ar_loss = reg_loss(z_tilde, attributes, len(data),
                              gamma=ar_weight, factor=config['factor'])
            loss += ar_loss

        # L1 weight regularization
        if config['is_L1']:
            l1_crit = nn.L1Loss(reduction="sum")
            weight_reg = sum(l1_crit(p, torch.zeros_like(p)) for p in model.parameters())
            loss += 0.00005 * weight_reg

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc, auc = safe_mean_accuracy(label, out_mlp)

        # Accumulate
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss2.item()
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
    config: Dict[str, Any]
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

            # Compute losses - using MEAN reduction for reconstruction
            recon_loss = reconstruction_loss_mean(recon_batch, data, config['recon_param'])
            mlp_loss = mlp_loss_function(label, out_mlp, config['alpha'])

            # Use full weights for validation (no warmup)
            kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, config['beta'], c=0.0)

            loss = recon_loss + mlp_loss + kl_loss2

            # Attribute regularization loss
            ar_loss = torch.tensor(0.0, device=device)
            if config['use_AR_LOSS']:
                ar_loss = reg_loss(z_tilde, attributes, len(data),
                                  gamma=config['gamma'], factor=config['factor'])
                loss += ar_loss

            # Compute accuracy
            acc, auc = safe_mean_accuracy(label, out_mlp)

            # Accumulate
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss2.item()
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


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save configuration
    try:
        import yaml
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except ImportError:
        import json
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    data = np.load(config['data_file'], allow_pickle=True).tolist()
    print(f"Loaded {len(data)} patients")

    # Create patient extractor
    extractor = create_patient_extractor(config['spreadsheet'])

    # Compute age stats
    age_stats = compute_age_stats(data, extractor)
    print(f"Age stats: mean={age_stats['mean']:.1f}, std={age_stats['std']:.1f}")

    # Train/val split
    train_data, val_data = create_stratified_split(
        data,
        val_fraction=config['val_fraction'],
        seed=config['seed']
    )

    # Create datasets
    print("\nCreating datasets...")
    train_ds = AttriVAEDataset(
        train_data, extractor,
        target_size=tuple(config['target_size']),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=config['augment']
    )
    val_ds = AttriVAEDataset(
        val_data, extractor,
        target_size=tuple(config['target_size']),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False
    )

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)

    model = ConvVAE(
        image_channels=config['image_channels'],
        h_dim=config['h_dim'],
        latent_size=config['latent_size'],
        n_filters_ENC=config['n_filters_enc'],
        n_filters_DEC=config['n_filters_dec']
    ).to(device)
    model.apply(initialize_weights)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0

    if args.resume:
        resume_path = output_dir.parent / 'latest' / 'checkpoint.pth'
        if resume_path.exists():
            print(f"\nResuming from {resume_path}")
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(f"No checkpoint found at {resume_path}")

    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Use AR Loss: {config['use_AR_LOSS']}")
    print(f"AR gamma: {config['gamma']}, factor: {config['factor']}")

    history = {'train': [], 'val': []}

    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, config)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(epoch, model, val_loader, config)
        history['val'].append(val_metrics)

        # Print progress
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, AR: {train_metrics['ar_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, AR: {val_metrics['ar_loss']:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        save_checkpoint(checkpoint, output_dir, 'checkpoint.pth')

        # Save best models
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(checkpoint, output_dir, 'best_loss_model.pth')
            print(f"  -> New best loss: {best_val_loss:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(checkpoint, output_dir, 'best_acc_model.pth')
            print(f"  -> New best accuracy: {best_val_acc:.4f}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            save_checkpoint(checkpoint, output_dir, 'best_auc_model.pth')
            print(f"  -> New best AUC: {best_val_auc:.4f}")

    # Save final model and history
    save_checkpoint(checkpoint, output_dir, 'final_model.pth')

    np.save(output_dir / 'history.npy', history)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')

    args = parser.parse_args()
    main(args)
