#!/usr/bin/env python3
"""
Visualization utilities for VQ-VAE pre-training results.

This script provides:
1. Original vs reconstructed slice comparisons
2. Training loss curves (VQ-VAE specific)
3. Codebook usage analysis
4. Train vs validation comparison for overfitting diagnosis

Note: VQ-VAE uses discrete codes, so latent interpolation doesn't work
as smoothly as with standard VAEs.

Usage:
    python visualize_vqvae.py --model outputs/pretrain_vqvae/TIMESTAMP/best_recon_model.pth

Author: Claude
Date: 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
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

from torch.utils.data import DataLoader

from model_vqvae_2d import VQVAE2D
from dataset_2d import (
    AttriVAEDataset2D,
    create_patient_extractor,
    create_stratified_split,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_config(model_path: Path) -> Tuple[VQVAE2D, Dict[str, Any]]:
    """Load trained model and its configuration.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Get hidden_dims from config
    hidden_dims = config.get('hidden_dims', [128, 256])

    # Legacy fix: old checkpoints saved hidden_dims after it was reversed in-place
    # during model construction. Detect this by checking if dims are in descending order.
    # Normal config should be ascending (e.g., [128, 256]), reversed is descending ([256, 128]).
    if len(hidden_dims) >= 2 and hidden_dims[0] > hidden_dims[-1]:
        # Reverse back to get the original order
        hidden_dims = hidden_dims[::-1]

    # Create model
    model = VQVAE2D(
        in_channels=config.get('image_channels', 1),
        embedding_dim=config.get('embedding_dim', 64),
        num_embeddings=config.get('num_embeddings', 512),
        hidden_dims=hidden_dims,
        beta=config.get('beta', 0.25),
        img_size=config.get('img_size', 80),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def plot_reconstructions(
    model: VQVAE2D,
    dataloader: DataLoader,
    n_samples: int = 8,
    title: str = "Reconstructions",
    save_path: Optional[Path] = None,
):
    """Plot original vs reconstructed slices.

    Args:
        model: Trained model
        dataloader: Data loader
        n_samples: Number of samples to show
        title: Title for the figure
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Get a batch of data
    data, labels, attributes, subject_ids, slice_indices = next(iter(dataloader))
    data = data[:n_samples].to(device)
    labels = labels[:n_samples]

    # Get reconstructions
    with torch.no_grad():
        recon, _, vq_loss = model(data)

    # Convert to numpy
    data_np = data.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # Calculate average MSE
    avg_mse = np.mean((data_np - recon_np) ** 2)

    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nLabel: {labels[i].item()}', fontsize=8)
        axes[0, i].axis('off')

        # Reconstructed
        axes[1, i].imshow(recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        mse = np.mean((data_np[i, 0] - recon_np[i, 0]) ** 2)
        axes[1, i].set_title(f'Recon\nMSE: {mse:.4f}', fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle(f'{title} (Avg MSE: {avg_mse:.6f})', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstructions to {save_path}")
    else:
        plt.show()

    plt.close()

    return avg_mse


def plot_train_val_comparison(
    model: VQVAE2D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_samples: int = 6,
    save_path: Optional[Path] = None,
):
    """Plot training vs validation reconstructions side by side for overfitting diagnosis.

    This helps diagnose:
    - Overfitting: Train recon good, Val recon bad
    - Underfitting: Both train and val recon bad
    - Good fit: Both train and val recon good

    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_samples: Number of samples to show per set
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Get training data
    train_data, train_labels, _, _, _ = next(iter(train_loader))
    train_data = train_data[:n_samples].to(device)
    train_labels = train_labels[:n_samples]

    # Get validation data
    val_data, val_labels, _, _, _ = next(iter(val_loader))
    val_data = val_data[:n_samples].to(device)
    val_labels = val_labels[:n_samples]

    # Get reconstructions
    with torch.no_grad():
        train_recon, _, _ = model(train_data)
        val_recon, _, _ = model(val_data)

    # Convert to numpy
    train_data_np = train_data.cpu().numpy()
    train_recon_np = train_recon.cpu().numpy()
    val_data_np = val_data.cpu().numpy()
    val_recon_np = val_recon.cpu().numpy()

    # Calculate MSEs
    train_mses = [np.mean((train_data_np[i, 0] - train_recon_np[i, 0]) ** 2) for i in range(n_samples)]
    val_mses = [np.mean((val_data_np[i, 0] - val_recon_np[i, 0]) ** 2) for i in range(n_samples)]
    avg_train_mse = np.mean(train_mses)
    avg_val_mse = np.mean(val_mses)

    # Create figure: 4 rows (train orig, train recon, val orig, val recon) x n_samples cols
    fig, axes = plt.subplots(4, n_samples, figsize=(2.5 * n_samples, 10))

    # Training samples
    for i in range(n_samples):
        # Train original
        axes[0, i].imshow(train_data_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Label: {train_labels[i].item()}', fontsize=8)
        axes[0, i].axis('off')

        # Train reconstructed
        axes[1, i].imshow(train_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'MSE: {train_mses[i]:.4f}', fontsize=8)
        axes[1, i].axis('off')

        # Val original
        axes[2, i].imshow(val_data_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Label: {val_labels[i].item()}', fontsize=8)
        axes[2, i].axis('off')

        # Val reconstructed
        axes[3, i].imshow(val_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title(f'MSE: {val_mses[i]:.4f}', fontsize=8)
        axes[3, i].axis('off')

    # Add row labels
    fig.text(0.02, 0.875, 'TRAIN\nOriginal', va='center', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.625, 'TRAIN\nRecon', va='center', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.375, 'VAL\nOriginal', va='center', ha='center', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.125, 'VAL\nRecon', va='center', ha='center', fontsize=10, fontweight='bold')

    # Diagnosis text
    gap = avg_val_mse - avg_train_mse
    gap_pct = (gap / avg_train_mse * 100) if avg_train_mse > 0 else 0

    if gap_pct > 50:
        diagnosis = "OVERFITTING: Val MSE >> Train MSE"
        color = 'red'
    elif avg_train_mse > 0.01 and avg_val_mse > 0.01:
        diagnosis = "UNDERFITTING: Both MSEs high"
        color = 'orange'
    else:
        diagnosis = "GOOD FIT: Both MSEs low and similar"
        color = 'green'

    plt.suptitle(
        f'VQ-VAE Train vs Validation Reconstruction Comparison\n'
        f'Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f} | Gap: {gap_pct:.1f}%\n'
        f'{diagnosis}',
        fontsize=12,
        color=color if 'OVERFITTING' in diagnosis or 'UNDERFITTING' in diagnosis else 'black'
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.88)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved train/val comparison to {save_path}")
    else:
        plt.show()

    plt.close()

    return avg_train_mse, avg_val_mse


def plot_training_curves(
    history_path: Path,
    save_path: Optional[Path] = None,
):
    """Plot training and validation loss curves for VQ-VAE.

    Args:
        history_path: Path to history.npy file
        save_path: Path to save figure (optional)
    """
    history = np.load(history_path, allow_pickle=True).item()

    train_metrics = history['train']
    val_metrics = history['val']

    epochs = range(1, len(train_metrics) + 1)

    # Extract metrics
    train_loss = [m['loss'] for m in train_metrics]
    train_recon = [m['recon_loss'] for m in train_metrics]
    train_vq = [m['vq_loss'] for m in train_metrics]
    train_ssim = [m.get('ssim_loss', 0) for m in train_metrics]

    val_loss = [m['loss'] for m in val_metrics]
    val_recon = [m['recon_loss'] for m in val_metrics]
    val_vq = [m['vq_loss'] for m in val_metrics]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].plot(epochs, train_loss, label='Train', alpha=0.8)
    axes[0, 0].plot(epochs, val_loss, label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(epochs, train_recon, label='Train', alpha=0.8)
    axes[0, 1].plot(epochs, val_recon, label='Val', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Reconstruction Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # VQ loss
    axes[1, 0].plot(epochs, train_vq, label='Train', alpha=0.8)
    axes[1, 0].plot(epochs, val_vq, label='Val', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('VQ Loss')
    axes[1, 0].set_title('Vector Quantization Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # SSIM loss (if used)
    if any(s > 0 for s in train_ssim):
        axes[1, 1].plot(epochs, train_ssim, label='Train', alpha=0.8, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM Loss')
        axes[1, 1].set_title('SSIM Loss (1 - SSIM)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show train vs val gap over time
        gap = [(v - t) / t * 100 if t > 0 else 0 for t, v in zip(train_recon, val_recon)]
        axes[1, 1].plot(epochs, gap, color='orange', alpha=0.8)
        axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No gap')
        axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% gap')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (%)')
        axes[1, 1].set_title('Train-Val Gap (Overfitting Indicator)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_codebook_usage(
    model: VQVAE2D,
    dataloader: DataLoader,
    save_path: Optional[Path] = None,
):
    """Analyze and plot codebook usage.

    This shows:
    - How many of the codebook entries are being used
    - Distribution of codebook usage

    Args:
        model: Trained model
        dataloader: Data loader
        save_path: Path to save figure (optional)
    """
    model.eval()

    all_indices = []

    with torch.no_grad():
        for data, _, _, _, _ in dataloader:
            data = data.to(device)
            indices = model.get_codebook_indices(data)
            all_indices.append(indices.cpu().numpy().flatten())

    # Concatenate all indices
    all_indices = np.concatenate(all_indices)

    # Count usage
    num_embeddings = model.num_embeddings
    counts = np.bincount(all_indices, minlength=num_embeddings)

    # Statistics
    used_codes = np.sum(counts > 0)
    usage_pct = used_codes / num_embeddings * 100

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of usage counts
    axes[0].bar(range(num_embeddings), counts, width=1.0, alpha=0.7)
    axes[0].set_xlabel('Codebook Index')
    axes[0].set_ylabel('Usage Count')
    axes[0].set_title(f'Codebook Usage Distribution\n{used_codes}/{num_embeddings} codes used ({usage_pct:.1f}%)')
    axes[0].grid(True, alpha=0.3)

    # Sorted usage (to see distribution better)
    sorted_counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100

    ax2 = axes[1]
    ax2.bar(range(len(sorted_counts)), sorted_counts, width=1.0, alpha=0.7, label='Count')
    ax2.set_xlabel('Codebook Index (sorted by usage)')
    ax2.set_ylabel('Usage Count')
    ax2.set_title('Sorted Codebook Usage')
    ax2.grid(True, alpha=0.3)

    # Add cumulative percentage line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(cumsum)), cumsum, 'r-', alpha=0.8, label='Cumulative %')
    ax2_twin.set_ylabel('Cumulative %', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.axhline(y=90, color='red', linestyle='--', alpha=0.5)
    ax2_twin.set_ylim(0, 105)

    # Find how many codes cover 90% of usage
    codes_for_90 = np.searchsorted(cumsum, 90) + 1
    ax2.axvline(x=codes_for_90, color='green', linestyle='--', alpha=0.7,
                label=f'90% coverage: {codes_for_90} codes')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved codebook usage to {save_path}")
    else:
        plt.show()

    plt.close()

    return used_codes, usage_pct


def plot_codebook_visualization(
    model: VQVAE2D,
    dataloader: DataLoader,
    n_samples: int = 4,
    save_path: Optional[Path] = None,
):
    """Visualize which codebook indices are used in different spatial locations.

    Args:
        model: Trained model
        dataloader: Data loader
        n_samples: Number of samples to show
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Get a batch of data
    data, labels, _, _, _ = next(iter(dataloader))
    data = data[:n_samples].to(device)

    # Get codebook indices
    with torch.no_grad():
        recon, _, _ = model(data)
        indices = model.get_codebook_indices(data)

    # Convert to numpy
    data_np = data.cpu().numpy()
    recon_np = recon.cpu().numpy()
    indices_np = indices.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original', fontsize=10)
        axes[0, i].axis('off')

        # Codebook indices
        im = axes[1, i].imshow(indices_np[i], cmap='tab20', vmin=0, vmax=model.num_embeddings)
        axes[1, i].set_title(f'Codebook Indices\n(unique: {len(np.unique(indices_np[i]))})', fontsize=10)
        axes[1, i].axis('off')

        # Reconstruction
        axes[2, i].imshow(recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        mse = np.mean((data_np[i, 0] - recon_np[i, 0]) ** 2)
        axes[2, i].set_title(f'Reconstruction\nMSE: {mse:.4f}', fontsize=10)
        axes[2, i].axis('off')

    plt.suptitle('VQ-VAE Codebook Index Visualization', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved codebook visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main(args):
    """Main visualization function."""
    model_path = Path(args.model)
    output_dir = model_path.parent

    print(f"Loading model from {model_path}")
    model, config = load_model_and_config(model_path)

    # Load data
    print("Loading data...")
    data = np.load(config['data_file'], allow_pickle=True).tolist()
    extractor = create_patient_extractor(config['spreadsheet'])
    age_stats = compute_age_stats(data, extractor)

    # Get both train and val splits
    train_data, val_data = create_stratified_split(
        data,
        val_fraction=config.get('val_fraction', 0.2),
        seed=config.get('seed', 42),
    )

    target_size = (config.get('img_size', 80), config.get('img_size', 80))

    # Create training dataset (without augmentation for fair comparison)
    train_dataset = AttriVAEDataset2D(
        train_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )

    val_dataset = AttriVAEDataset2D(
        val_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=0,
    )

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Train vs Val comparison (most important for diagnosing overfitting/underfitting)
    print("\n[1/6] Train vs Val comparison...")
    plot_train_val_comparison(
        model, train_loader, val_loader, n_samples=6,
        save_path=output_dir / 'train_val_comparison.png'
    )

    # 2. Training reconstructions only
    print("[2/6] Training reconstructions...")
    plot_reconstructions(
        model, train_loader, n_samples=8,
        title="Training Set Reconstructions",
        save_path=output_dir / 'reconstructions_train.png'
    )

    # 3. Validation reconstructions only
    print("[3/6] Validation reconstructions...")
    plot_reconstructions(
        model, val_loader, n_samples=8,
        title="Validation Set Reconstructions",
        save_path=output_dir / 'reconstructions_val.png'
    )

    # 4. Training curves
    print("[4/6] Training curves...")
    history_path = output_dir / 'history.npy'
    if history_path.exists():
        plot_training_curves(
            history_path,
            save_path=output_dir / 'training_curves.png'
        )

    # 5. Codebook usage
    print("[5/6] Codebook usage analysis...")
    used_codes, usage_pct = plot_codebook_usage(
        model, val_loader,
        save_path=output_dir / 'codebook_usage.png'
    )
    print(f"       {used_codes}/{model.num_embeddings} codes used ({usage_pct:.1f}%)")

    # 6. Codebook visualization
    print("[6/6] Codebook index visualization...")
    plot_codebook_visualization(
        model, val_loader, n_samples=4,
        save_path=output_dir / 'codebook_visualization.png'
    )

    print(f"\nAll visualizations saved to {output_dir}")
    print("\nKey files for diagnosing quality:")
    print(f"  - {output_dir / 'train_val_comparison.png'} (side-by-side comparison)")
    print(f"  - {output_dir / 'training_curves.png'} (loss curves)")
    print(f"  - {output_dir / 'codebook_usage.png'} (codebook utilization)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE results")
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    args = parser.parse_args()
    main(args)
