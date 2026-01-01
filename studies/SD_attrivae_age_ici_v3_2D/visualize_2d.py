#!/usr/bin/env python3
"""
Visualization utilities for 2D VAE pre-training results.

This script provides:
1. Original vs reconstructed slice comparisons
2. Training loss curves
3. Latent space visualization (t-SNE/PCA)
4. Random generation from latent space

Usage:
    python visualize_2d.py --model outputs/pretrain_2d/TIMESTAMP/best_recon_model.pth

Author: Claude
Date: 2025-12-31
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from torch.utils.data import DataLoader

from model_2d import ConvVAE2D
from dataset_2d import (
    AttriVAEDataset2D,
    create_patient_extractor,
    create_stratified_split,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_config(model_path: Path) -> Tuple[ConvVAE2D, Dict[str, Any]]:
    """Load trained model and its configuration.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = ConvVAE2D(
        image_channels=config.get('image_channels', 1),
        h_dim=config.get('h_dim', 64),
        latent_size=config.get('latent_size', 16),
        n_filters_ENC=config.get('n_filters_enc', [8, 16, 32, 64, 2]),
        n_filters_DEC=config.get('n_filters_dec', [64, 32, 16, 8, 4, 2]),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def plot_reconstructions(
    model: ConvVAE2D,
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
        recon, mu, logvar, _, _, _, _, _, _ = model(data)

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
    model: ConvVAE2D,
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
        train_recon, _, _, _, _, _, _, _, _ = model(train_data)
        val_recon, _, _, _, _, _, _, _, _ = model(val_data)

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
        if i == 0:
            axes[0, i].set_ylabel('Train\nOriginal', fontsize=10)

        # Train reconstructed
        axes[1, i].imshow(train_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'MSE: {train_mses[i]:.4f}', fontsize=8)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Train\nRecon', fontsize=10)

        # Val original
        axes[2, i].imshow(val_data_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Label: {val_labels[i].item()}', fontsize=8)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Val\nOriginal', fontsize=10)

        # Val reconstructed
        axes[3, i].imshow(val_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title(f'MSE: {val_mses[i]:.4f}', fontsize=8)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Val\nRecon', fontsize=10)

    # Add row labels manually since ylabel doesn't work well with imshow
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
        f'Train vs Validation Reconstruction Comparison\n'
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
    """Plot training and validation loss curves.

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
    train_kl = [m['kl_loss'] for m in train_metrics]
    train_beta = [m['beta'] for m in train_metrics]

    val_loss = [m['loss'] for m in val_metrics]
    val_recon = [m['recon_loss'] for m in val_metrics]
    val_kl = [m['kl_loss'] for m in val_metrics]

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

    # KL loss
    axes[1, 0].plot(epochs, train_kl, label='Train', alpha=0.8)
    axes[1, 0].plot(epochs, val_kl, label='Val', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Beta warmup
    axes[1, 1].plot(epochs, train_beta, color='green', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].set_title('Beta (KL Weight) Warmup')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_space(
    model: ConvVAE2D,
    dataloader: DataLoader,
    method: str = 'pca',
    color_by: str = 'label',
    save_path: Optional[Path] = None,
):
    """Visualize latent space using PCA or t-SNE.

    Args:
        model: Trained model
        dataloader: Data loader
        method: 'pca' or 'tsne'
        color_by: 'label', 'age', or 'ici'
        save_path: Path to save figure (optional)
    """
    model.eval()

    all_z = []
    all_labels = []
    all_ages = []
    all_ici = []

    with torch.no_grad():
        for data, labels, attributes, subject_ids, slice_indices in dataloader:
            data = data.to(device)
            mu, logvar, z_dist = model.encode(data)
            all_z.append(mu.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ages.extend(attributes[:, 0].numpy())  # Normalized age
            all_ici.extend(attributes[:, 1].numpy())

    # Concatenate
    z = np.concatenate(all_z, axis=0)
    labels = np.array(all_labels)
    ages = np.array(all_ages)
    ici = np.array(all_ici)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        z_2d = reducer.fit_transform(z)
        explained_var = reducer.explained_variance_ratio_
        title_suffix = f'(explained var: {explained_var[0]:.2%}, {explained_var[1]:.2%})'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z) - 1))
        z_2d = reducer.fit_transform(z)
        title_suffix = ''

    # Select color values
    if color_by == 'label':
        colors = labels
        cmap = 'coolwarm'
        label = 'CHIP Label'
    elif color_by == 'age':
        colors = ages
        cmap = 'viridis'
        label = 'Age (normalized)'
    else:
        colors = ici
        cmap = 'coolwarm'
        label = 'ICI Treatment'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, cmap=cmap, alpha=0.6, s=20)
    plt.colorbar(scatter, label=label)

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Latent Space Visualization ({method.upper()}) {title_suffix}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent space plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_random_samples(
    model: ConvVAE2D,
    n_samples: int = 16,
    save_path: Optional[Path] = None,
):
    """Generate random samples from the prior.

    Args:
        model: Trained model
        n_samples: Number of samples to generate
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Sample from prior (standard normal)
    z = torch.randn(n_samples, model.latent_size).to(device)

    # Decode
    with torch.no_grad():
        samples = model.decode(z)

    # Convert to numpy
    samples_np = samples.cpu().numpy()

    # Create figure
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(samples_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Random Samples from Prior', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved random samples to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_interpolation(
    model: ConvVAE2D,
    dataloader: DataLoader,
    n_steps: int = 10,
    save_path: Optional[Path] = None,
):
    """Interpolate between two samples in latent space.

    Args:
        model: Trained model
        dataloader: Data loader (to get two real samples)
        n_steps: Number of interpolation steps
        save_path: Path to save figure (optional)
    """
    model.eval()

    # Get two samples
    data, labels, _, _, _ = next(iter(dataloader))
    data = data[:2].to(device)

    # Encode
    with torch.no_grad():
        mu1, _, _ = model.encode(data[0:1])
        mu2, _, _ = model.encode(data[1:2])

    # Interpolate
    alphas = torch.linspace(0, 1, n_steps).to(device)
    interpolated = []

    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            decoded = model.decode(z)
            interpolated.append(decoded.cpu().numpy()[0, 0])

    # Create figure
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))

    # Original 1
    axes[0].imshow(data[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original 1')
    axes[0].axis('off')

    # Interpolated
    for i, img in enumerate(interpolated):
        axes[i + 1].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'α={alphas[i]:.1f}')
        axes[i + 1].axis('off')

    # Original 2
    axes[-1].imshow(data[1, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[-1].set_title('Original 2')
    axes[-1].axis('off')

    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpolation to {save_path}")
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

    target_size = tuple(config.get('target_size', [80, 80]))

    # Create training dataset (without augmentation for fair comparison)
    train_dataset = AttriVAEDataset2D(
        train_data, extractor,
        target_size=target_size,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,  # No augmentation for visualization
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
        shuffle=True,  # Shuffle to get random samples
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
    print("\n[1/7] Train vs Val comparison...")
    plot_train_val_comparison(
        model, train_loader, val_loader, n_samples=6,
        save_path=output_dir / 'train_val_comparison.png'
    )

    # 2. Training reconstructions only
    print("[2/7] Training reconstructions...")
    plot_reconstructions(
        model, train_loader, n_samples=8,
        title="Training Set Reconstructions",
        save_path=output_dir / 'reconstructions_train.png'
    )

    # 3. Validation reconstructions only
    print("[3/7] Validation reconstructions...")
    plot_reconstructions(
        model, val_loader, n_samples=8,
        title="Validation Set Reconstructions",
        save_path=output_dir / 'reconstructions_val.png'
    )

    # 4. Training curves
    print("[4/7] Training curves...")
    history_path = output_dir / 'history.npy'
    if history_path.exists():
        plot_training_curves(
            history_path,
            save_path=output_dir / 'training_curves.png'
        )

    # 5. Latent space (PCA)
    print("[5/7] Latent space PCA...")
    plot_latent_space(
        model, val_loader, method='pca', color_by='label',
        save_path=output_dir / 'latent_pca_label.png'
    )

    # 6. Random samples
    print("[6/7] Random samples from prior...")
    plot_random_samples(
        model, n_samples=16,
        save_path=output_dir / 'random_samples.png'
    )

    # 7. Interpolation
    print("[7/7] Latent space interpolation...")
    plot_latent_interpolation(
        model, val_loader, n_steps=8,
        save_path=output_dir / 'interpolation.png'
    )

    print(f"\nAll visualizations saved to {output_dir}")
    print("\nKey files for diagnosing overfitting/underfitting:")
    print(f"  - {output_dir / 'train_val_comparison.png'} (side-by-side comparison)")
    print(f"  - {output_dir / 'training_curves.png'} (loss curves)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 2D VAE results")
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    args = parser.parse_args()
    main(args)
