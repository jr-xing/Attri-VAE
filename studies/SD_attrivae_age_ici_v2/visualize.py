#!/usr/bin/env python3
"""
Visualization script for AttriVAE v2 (two-stage training).

Generates:
1. Reconstruction comparison (original vs. reconstructed)
2. Latent space distribution with disentanglement correlation
3. Age traversal (vary dim 0)
4. ICI traversal (vary dim 1)
5. 2D Age x ICI grid

Usage:
    python visualize.py --checkpoint outputs/finetune/<timestamp>/best_auc_model.pth

Author: Claude
Date: 2025-12-28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
    print(f"[Model] Loaded from {checkpoint_path}")
    return model


def encode_image(model: ConvVAE, image: torch.Tensor) -> torch.Tensor:
    """Encode image to latent vector (using mean)."""
    if image.dim() == 4:
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        mu, logvar, z_dist = model.encode(image)
    return mu


def decode_latent(model: ConvVAE, z: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to image."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.to(device)

    with torch.no_grad():
        recon = model.decode(z)
    return recon


def visualize_reconstruction(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    n_samples: int = 4,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare original vs reconstructed images."""
    model.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i, idx in enumerate(indices):
        image, label, attributes, subject_id = dataset[idx]

        # Original - slice along D dimension
        orig_slice = image[0, slice_idx, :, :].numpy()

        # Reconstruction
        z = encode_image(model, image)
        recon = decode_latent(model, z)
        recon_slice = recon[0, 0, slice_idx, :, :].cpu().numpy()

        # Get raw attributes
        raw_attrs = dataset.get_raw_attributes(idx)
        age = raw_attrs.get('age', 'N/A')
        ici = 'Yes' if raw_attrs.get('ici') == 1 else 'No'
        chip = 'CHIP+' if label == 1 else 'CHIP-'

        # Plot original
        axes[0, i].imshow(orig_slice, cmap='gray')
        axes[0, i].set_title(f'{subject_id}\nAge:{age}, ICI:{ici}, {chip}', fontsize=9)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Plot reconstruction
        axes[1, i].imshow(recon_slice, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruction', fontsize=12)

    fig.suptitle('Original vs. Reconstruction', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_age_traversal(
    model: ConvVAE,
    image: torch.Tensor,
    age_mean: float = 63.7,
    age_std: float = 14.0,
    n_steps: int = 9,
    age_range: Tuple[float, float] = (30, 90),
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize age traversal (dimension 0)."""
    z_base = encode_image(model, image)

    ages_years = np.linspace(age_range[0], age_range[1], n_steps)
    ages_normalized = (ages_years - age_mean) / age_std

    fig, axes = plt.subplots(1, n_steps, figsize=(2.5 * n_steps, 3))

    for i, (age_years, age_norm) in enumerate(zip(ages_years, ages_normalized)):
        z_modified = z_base.clone()
        z_modified[0, 0] = age_norm

        with torch.no_grad():
            recon = model.decode(z_modified)

        img_np = recon[0, 0, slice_idx, :, :].cpu().numpy()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'{int(age_years)}', fontsize=10)
        axes[i].axis('off')

    fig.suptitle('Age Traversal (Latent Dim 0)', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_ici_traversal(
    model: ConvVAE,
    image: torch.Tensor,
    n_steps: int = 5,
    ici_range: Tuple[float, float] = (-1.0, 2.0),
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize ICI traversal (dimension 1)."""
    z_base = encode_image(model, image)

    ici_values = np.linspace(ici_range[0], ici_range[1], n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(2.5 * n_steps, 3))

    for i, ici_val in enumerate(ici_values):
        z_modified = z_base.clone()
        z_modified[0, 1] = ici_val

        with torch.no_grad():
            recon = model.decode(z_modified)

        img_np = recon[0, 0, slice_idx, :, :].cpu().numpy()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'ICI={ici_val:.1f}', fontsize=10)
        axes[i].axis('off')

    fig.suptitle('ICI Traversal (Latent Dim 1)', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def visualize_2d_grid(
    model: ConvVAE,
    image: torch.Tensor,
    age_mean: float = 63.7,
    age_std: float = 14.0,
    age_steps: int = 5,
    ici_steps: int = 3,
    age_range: Tuple[float, float] = (40, 85),
    ici_range: Tuple[float, float] = (0.0, 1.0),
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize 2D grid of Age x ICI traversal."""
    z_base = encode_image(model, image)

    ages_years = np.linspace(age_range[0], age_range[1], age_steps)
    ages_normalized = (ages_years - age_mean) / age_std
    ici_values = np.linspace(ici_range[0], ici_range[1], ici_steps)

    fig, axes = plt.subplots(ici_steps, age_steps, figsize=(2.5 * age_steps, 2.5 * ici_steps))

    if ici_steps == 1:
        axes = axes.reshape(1, -1)
    if age_steps == 1:
        axes = axes.reshape(-1, 1)

    for i, ici_val in enumerate(ici_values):
        for j, (age_years, age_norm) in enumerate(zip(ages_years, ages_normalized)):
            z_modified = z_base.clone()
            z_modified[0, 0] = age_norm
            z_modified[0, 1] = ici_val

            with torch.no_grad():
                recon = model.decode(z_modified)

            img_np = recon[0, 0, slice_idx, :, :].cpu().numpy()

            ax = axes[i, j]
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')

            if i == 0:
                ax.set_title(f'Age {int(age_years)}', fontsize=10)
            if j == 0:
                ici_label = "No ICI" if ici_val < 0.5 else "ICI"
                ax.set_ylabel(ici_label, fontsize=10)

    fig.suptitle('Age × ICI Latent Space Manipulation', fontsize=14)
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
    """Visualize latent space distribution with disentanglement correlation."""
    model.eval()

    all_z = []
    all_ages = []
    all_ici = []
    all_labels = []

    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for idx in indices:
        image, label, attributes, subject_id = dataset[idx]
        z = encode_image(model, image)

        all_z.append(z.cpu().numpy())
        all_ages.append(attributes[0].item())
        all_ici.append(attributes[1].item())
        all_labels.append(label)

    all_z = np.concatenate(all_z, axis=0)
    all_ages = np.array(all_ages)
    all_ici = np.array(all_ici)
    all_labels = np.array(all_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Dim 0 vs Age
    ax = axes[0, 0]
    scatter = ax.scatter(all_z[:, 0], all_ages, c=all_labels, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Latent Dim 0 (Age)', fontsize=11)
    ax.set_ylabel('Actual Age (normalized)', fontsize=11)
    ax.set_title('Age Disentanglement', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='CHIP Label')
    corr = np.corrcoef(all_z[:, 0], all_ages)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    # Dim 1 vs ICI
    ax = axes[0, 1]
    ici_jittered = all_ici + np.random.normal(0, 0.05, len(all_ici))
    scatter = ax.scatter(all_z[:, 1], ici_jittered, c=all_labels, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Latent Dim 1 (ICI)', fontsize=11)
    ax.set_ylabel('Actual ICI (jittered)', fontsize=11)
    ax.set_title('ICI Disentanglement', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='CHIP Label')
    corr = np.corrcoef(all_z[:, 1], all_ici)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    # Latent space colored by age
    ax = axes[1, 0]
    scatter = ax.scatter(all_z[:, 0], all_z[:, 1], c=all_ages, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Latent Dim 0 (Age)', fontsize=11)
    ax.set_ylabel('Latent Dim 1 (ICI)', fontsize=11)
    ax.set_title('Latent Space (colored by Age)', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Age (norm)')

    # Latent space colored by ICI
    ax = axes[1, 1]
    scatter = ax.scatter(all_z[:, 0], all_z[:, 1], c=all_ici, cmap='RdYlBu', alpha=0.7)
    ax.set_xlabel('Latent Dim 0 (Age)', fontsize=11)
    ax.set_ylabel('Latent Dim 1 (ICI)', fontsize=11)
    ax.set_title('Latent Space (colored by ICI)', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='ICI')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize AttriVAE v2 results")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
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
        help='Output directory for visualizations'
    )
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--slice_idx', type=int, default=40, help='Slice index (0-79)')

    args = parser.parse_args()

    # Auto-detect config from checkpoint directory
    checkpoint_dir = Path(args.checkpoint).parent
    if args.config is None:
        config_path = checkpoint_dir / 'config.yaml'
        if not config_path.exists():
            config_path = STUDY_DIR / 'configs' / 'finetune.yaml'
    else:
        config_path = Path(args.config)

    # Load config
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

    # Default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(STUDY_DIR / 'outputs' / 'visualizations' / timestamp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, flat_config)

    # Load dataset
    print("[Data] Loading dataset...")
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

    # Sample indices
    sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)

    # Generate visualizations
    print("\n[Viz] Generating visualizations...")

    # 1. Reconstruction comparison
    visualize_reconstruction(
        model, dataset, n_samples=args.n_samples, slice_idx=args.slice_idx,
        output_path=str(output_dir / "reconstruction_comparison.png")
    )

    # 2. Latent distribution
    visualize_latent_distribution(
        model, dataset, max_samples=100,
        output_path=str(output_dir / "latent_distribution.png")
    )

    # 3. Age traversal
    for i, idx in enumerate(sample_indices[:2]):
        image, _, _, subj_id = dataset[idx]
        visualize_age_traversal(
            model, image, age_mean=age_stats['mean'], age_std=age_stats['std'],
            slice_idx=args.slice_idx,
            output_path=str(output_dir / f"age_traversal_{i}_{subj_id}.png")
        )

    # 4. ICI traversal
    for i, idx in enumerate(sample_indices[:2]):
        image, _, _, subj_id = dataset[idx]
        visualize_ici_traversal(
            model, image, slice_idx=args.slice_idx,
            output_path=str(output_dir / f"ici_traversal_{i}_{subj_id}.png")
        )

    # 5. 2D grid
    image, _, _, subj_id = dataset[sample_indices[0]]
    visualize_2d_grid(
        model, image, age_mean=age_stats['mean'], age_std=age_stats['std'],
        slice_idx=args.slice_idx,
        output_path=str(output_dir / f"age_ici_grid_{subj_id}.png")
    )

    print(f"\n[Done] All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
