#!/usr/bin/env python3
"""
Check pre-trained VAE reconstruction and sampling ability.

This script evaluates the Stage 1 pre-trained model:
1. Reconstruction quality (original vs reconstructed)
2. Random sampling from latent space
3. Latent space interpolation between samples
4. Reconstruction metrics (MSE, SSIM)

Usage:
    python check_pretrain.py --checkpoint outputs/pretrain/20251231_154957/best_recon_model.pth

Author: Claude
Date: 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
V1_STUDY_DIR = STUDY_DIR.parent / "SD_attrivae_age_ici"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(V1_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(V1_STUDY_DIR))

from model.model import ConvVAE
from dataset import (
    AttriVAEDataset,
    create_patient_extractor,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, config: Dict) -> ConvVAE:
    """Load trained model from checkpoint."""
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

    model.to(device)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"[Model] Trained for {checkpoint['epoch'] + 1} epochs")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"[Model] Checkpoint metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:.6f}")

    return model


def compute_reconstruction_metrics(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    n_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Compute reconstruction metrics over dataset.

    Args:
        model: Trained VAE model
        dataset: Dataset to evaluate
        n_samples: Number of samples (None = all)

    Returns:
        Dict with MSE, MAE metrics
    """
    model.eval()

    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for idx in indices:
            image, _, _, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Encode and decode
            mu, logvar, z_dist = model.encode(image)
            recon = model.decode(mu)  # Use mean for deterministic reconstruction

            # Compute metrics
            mse = F.mse_loss(recon, image, reduction='mean').item()
            mae = F.l1_loss(recon, image, reduction='mean').item()

            total_mse += mse
            total_mae += mae

    return {
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
        'rmse': np.sqrt(total_mse / n_samples),
    }


def visualize_reconstruction(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    n_samples: int = 6,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize original vs reconstructed images.

    Shows multiple slices for each sample.
    """
    model.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    # Show 3 slices per sample: slice_idx-20, slice_idx, slice_idx+20
    slice_offsets = [-20, 0, 20]
    n_slices = len(slice_offsets)

    fig, axes = plt.subplots(n_samples, 2 * n_slices, figsize=(3 * n_slices * 2, 3 * n_samples))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label, _, subject_id = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            # Encode and decode
            mu, logvar, z_dist = model.encode(image_tensor)
            recon = model.decode(mu)

            # Compute per-sample MSE
            mse = F.mse_loss(recon, image_tensor, reduction='mean').item()

            for j, offset in enumerate(slice_offsets):
                s = slice_idx + offset
                s = max(0, min(79, s))  # Clamp to valid range

                # Original
                orig_slice = image[0, s, :, :].numpy()
                axes[i, j].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
                if i == 0:
                    axes[i, j].set_title(f'Original (slice {s})', fontsize=10)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(f'{subject_id}\nMSE={mse:.4f}', fontsize=9)

                # Reconstruction
                recon_slice = recon[0, 0, s, :, :].cpu().numpy()
                axes[i, n_slices + j].imshow(recon_slice, cmap='gray', vmin=0, vmax=1)
                if i == 0:
                    axes[i, n_slices + j].set_title(f'Recon (slice {s})', fontsize=10)
                axes[i, n_slices + j].axis('off')

    fig.suptitle('Pre-trained VAE: Original vs Reconstruction', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_random_samples(
    model: ConvVAE,
    n_samples: int = 16,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate random samples from the latent space.

    Samples z ~ N(0, I) and decodes to images.
    """
    model.eval()
    latent_size = model.latent_size

    # Sample from standard normal
    z = torch.randn(n_samples, latent_size).to(device)

    with torch.no_grad():
        samples = model.decode(z)

    # Arrange in grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_samples):
        img = samples[i, 0, slice_idx, :, :].cpu().numpy()
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=9)

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Random Samples from Latent Space z ~ N(0, I)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_interpolation(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    n_steps: int = 10,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize latent space interpolation between two samples."""
    model.eval()

    # Pick two random samples
    idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)

    image1, _, _, subj1 = dataset[idx1]
    image2, _, _, subj2 = dataset[idx2]

    image1 = image1.unsqueeze(0).to(device)
    image2 = image2.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode both
        mu1, _, _ = model.encode(image1)
        mu2, _, _ = model.encode(image2)

        # Interpolate
        alphas = np.linspace(0, 1, n_steps)
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z_interp)
            interpolated.append(recon[0, 0, slice_idx, :, :].cpu().numpy())

    # Plot
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(2.5 * (n_steps + 2), 3))

    # Original 1
    axes[0].imshow(image1[0, 0, slice_idx, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{subj1}\n(start)', fontsize=9)
    axes[0].axis('off')

    # Interpolations
    for i, img in enumerate(interpolated):
        axes[i + 1].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'α={alphas[i]:.1f}', fontsize=9)
        axes[i + 1].axis('off')

    # Original 2
    axes[-1].imshow(image2[0, 0, slice_idx, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[-1].set_title(f'{subj2}\n(end)', fontsize=9)
    axes[-1].axis('off')

    fig.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_latent_distribution(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    max_samples: int = 100,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize the distribution of latent codes."""
    model.eval()

    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    all_mu = []
    all_std = []

    with torch.no_grad():
        for idx in indices:
            image, _, _, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            mu, logvar, _ = model.encode(image)
            std = torch.exp(0.5 * logvar)

            all_mu.append(mu.cpu().numpy())
            all_std.append(std.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)  # (N, latent_size)
    all_std = np.concatenate(all_std, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean of mu across samples for each dimension
    ax = axes[0, 0]
    mu_mean = all_mu.mean(axis=0)
    mu_std = all_mu.std(axis=0)
    dims = np.arange(len(mu_mean))
    ax.bar(dims, mu_mean, yerr=mu_std, alpha=0.7, capsize=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean μ')
    ax.set_title('Mean of Latent Means (μ) per Dimension')

    # Mean of std across samples
    ax = axes[0, 1]
    std_mean = all_std.mean(axis=0)
    ax.bar(dims, std_mean, alpha=0.7, color='orange')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='N(0,1) std')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean σ')
    ax.set_title('Mean of Latent Stds (σ) per Dimension')
    ax.legend()

    # Distribution of first two dimensions
    ax = axes[1, 0]
    ax.scatter(all_mu[:, 0], all_mu[:, 1], alpha=0.5)
    ax.set_xlabel('Latent Dim 0')
    ax.set_ylabel('Latent Dim 1')
    ax.set_title('Latent Space (Dim 0 vs Dim 1)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Histogram of all mu values
    ax = axes[1, 1]
    ax.hist(all_mu.flatten(), bins=50, alpha=0.7, density=True)
    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r--', label='N(0,1)')
    ax.set_xlabel('μ value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of All Latent Means')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Check pre-trained VAE")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=str(STUDY_DIR / 'outputs' / 'pretrain' / '20251231_154957' / 'best_recon_model.pth'),
        help='Path to pre-trained checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument('--slice_idx', type=int, default=40, help='Slice index (0-79)')

    args = parser.parse_args()

    # Auto-detect config
    checkpoint_dir = Path(args.checkpoint).parent
    if args.config is None:
        config_path = checkpoint_dir / 'config.yaml'
        if not config_path.exists():
            config_path = STUDY_DIR / 'configs' / 'pretrain.yaml'
    else:
        config_path = Path(args.config)

    # Load config
    print(f"[Config] Loading from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten config
    flat_config = {}
    for section in ['data', 'model', 'training', 'loss']:
        if section in config:
            flat_config.update(config[section])
    for key in config:
        if key not in ['data', 'model', 'training', 'loss']:
            flat_config[key] = config[key]

    # Output directory
    if args.output_dir is None:
        args.output_dir = str(checkpoint_dir / 'check_pretrain')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving to {output_dir}")

    # Load model
    print(f"\n[Model] Loading from {args.checkpoint}")
    model = load_model(args.checkpoint, flat_config)

    # Load dataset
    print("\n[Data] Loading dataset...")
    data = np.load(flat_config['data_file'], allow_pickle=True).tolist()
    extractor = create_patient_extractor(flat_config['spreadsheet'])
    age_stats = compute_age_stats(data, extractor)

    dataset = AttriVAEDataset(
        data, extractor,
        target_size=tuple(flat_config.get('target_size', [80, 80, 80])),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )
    print(f"[Data] Loaded {len(dataset)} samples")

    # Compute metrics
    print("\n[Metrics] Computing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(model, dataset, n_samples=50)
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")

    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("Pre-trained VAE Reconstruction Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"N samples: 50\n\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v:.6f}\n")

    # Generate visualizations
    print("\n[Viz] Generating visualizations...")

    print("  - Reconstruction comparison...")
    visualize_reconstruction(
        model, dataset, n_samples=6, slice_idx=args.slice_idx,
        output_path=str(output_dir / "reconstruction.png")
    )

    print("  - Random samples...")
    visualize_random_samples(
        model, n_samples=16, slice_idx=args.slice_idx,
        output_path=str(output_dir / "random_samples.png")
    )

    print("  - Latent interpolation...")
    visualize_interpolation(
        model, dataset, n_steps=8, slice_idx=args.slice_idx,
        output_path=str(output_dir / "interpolation.png")
    )

    print("  - Latent distribution...")
    visualize_latent_distribution(
        model, dataset, max_samples=100,
        output_path=str(output_dir / "latent_distribution.png")
    )

    print(f"\n[Done] All outputs saved to {output_dir}")
    print("\nNext steps:")
    print(f"  1. Check reconstruction.png - originals should match reconstructions")
    print(f"  2. Check random_samples.png - samples should look like cardiac images")
    print(f"  3. Check interpolation.png - smooth transition between images")
    print(f"  4. If satisfied, proceed to Stage 2 fine-tuning:")
    print(f"     python finetune_attrivae.py --pretrained {args.checkpoint}")


if __name__ == "__main__":
    main()
