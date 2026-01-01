"""
Latent Space Manipulation / Attribute Scanning for AttriVAE.

This script implements the "latent space manipulation" described in the paper:
1. Encode an input image to get its latent vector Z
2. Modify specific dimensions (age=dim0, ICI=dim1)
3. Decode the modified latent vector to generate new images
4. Visualize how images change as attributes vary

Usage:
    python visualize_latent_traversal.py --checkpoint <path_to_model.pth> --output_dir <output_path>

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
from matplotlib.gridspec import GridSpec
import yaml

# Add paths
ATTRIVAE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ATTRIVAE_ROOT))

from model.model import ConvVAE
from dataset import (
    AttriVAEDataset,
    create_patient_extractor,
    compute_age_stats,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, config: Dict) -> ConvVAE:
    """Load trained AttriVAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        config: Model configuration dict

    Returns:
        Loaded model in eval mode
    """
    model = ConvVAE(
        image_channels=config.get('image_channels', 1),
        h_dim=config.get('h_dim', 96),
        latent_size=config.get('latent_size', 64),
        n_filters_ENC=config.get('n_filters_enc', [8, 16, 32, 64, 2]),
        n_filters_DEC=config.get('n_filters_dec', [64, 32, 16, 8, 4, 2]),
    )

    # weights_only=False needed for PyTorch 2.6+ when loading checkpoints with numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {checkpoint_path}")
    return model


def encode_image(model: ConvVAE, image: torch.Tensor) -> torch.Tensor:
    """Encode an image to its latent representation.

    Args:
        model: Trained VAE model
        image: Input image tensor (1, C, D, H, W) or (C, D, H, W)

    Returns:
        Latent vector z (using mean, not sampled)
    """
    if image.dim() == 4:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        mu, logvar, z_dist = model.encode(image)
        # Use mean (mu) for deterministic traversal
        z = mu

    return z


def decode_latent(model: ConvVAE, z: torch.Tensor) -> torch.Tensor:
    """Decode a latent vector to an image.

    Args:
        model: Trained VAE model
        z: Latent vector (1, latent_size) or (latent_size,)

    Returns:
        Decoded image tensor (1, C, D, H, W)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)

    z = z.to(device)

    with torch.no_grad():
        recon = model.decode(z)

    return recon


def traverse_latent_dimension(
    model: ConvVAE,
    z_base: torch.Tensor,
    dim: int,
    values: np.ndarray,
) -> List[torch.Tensor]:
    """Traverse a single latent dimension and generate images.

    This is the core "attribute scanning" operation from the paper.

    Args:
        model: Trained VAE model
        z_base: Base latent vector to modify
        dim: Dimension to traverse (0=age, 1=ICI for this study)
        values: Array of values to set for that dimension

    Returns:
        List of generated images (each is tensor of shape (1, C, D, H, W))
    """
    images = []
    z_base = z_base.to(device)

    for val in values:
        z_modified = z_base.clone()
        z_modified[0, dim] = val

        with torch.no_grad():
            recon = model.decode(z_modified)

        images.append(recon.cpu())

    return images


def compute_latent_statistics(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    max_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """Compute statistics of latent dimensions across the dataset.

    Args:
        model: Trained VAE model
        dataset: Dataset to compute statistics from
        max_samples: Maximum samples to use

    Returns:
        Dict with 'mean', 'std', 'min', 'max' for each latent dimension
    """
    model.eval()
    all_z = []

    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for idx in indices:
        image, label, attributes, subject_id = dataset[idx]
        z = encode_image(model, image)
        all_z.append(z.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)  # (N, latent_size)

    return {
        'mean': np.mean(all_z, axis=0),
        'std': np.std(all_z, axis=0),
        'min': np.min(all_z, axis=0),
        'max': np.max(all_z, axis=0),
    }


def visualize_single_traversal(
    images: List[torch.Tensor],
    values: np.ndarray,
    attribute_name: str,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize a single latent dimension traversal.

    Args:
        images: List of generated images
        values: Corresponding attribute values
        attribute_name: Name of the attribute (e.g., "Age", "ICI")
        slice_idx: Which slice to show from 3D volume
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(2.5 * n_images, 3))

    if n_images == 1:
        axes = [axes]

    for i, (img, val) in enumerate(zip(images, values)):
        # Extract middle slice from 3D volume
        # img shape: (1, C, D, H, W) -> take slice from D dimension to get (H, W)
        img_np = img[0, 0, slice_idx, :, :].numpy()

        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'{val:.2f}', fontsize=10)
        axes[i].axis('off')

    fig.suptitle(f'{attribute_name} Traversal (Latent Dim Manipulation)', fontsize=12)
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
    """Visualize age traversal (dimension 0).

    Generates images showing how the cardiac structure changes
    as age varies from young to old.

    Args:
        model: Trained VAE model
        image: Input image to encode
        age_mean: Mean age for normalization
        age_std: Std age for normalization
        n_steps: Number of steps in traversal
        age_range: (min_age, max_age) in years
        slice_idx: Slice to visualize
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
    # Encode the input image
    z_base = encode_image(model, image)

    # Create age values (in years) and convert to normalized values
    ages_years = np.linspace(age_range[0], age_range[1], n_steps)
    ages_normalized = (ages_years - age_mean) / age_std

    # Traverse age dimension (dim 0)
    images = traverse_latent_dimension(model, z_base, dim=0, values=ages_normalized)

    # Visualize
    fig = visualize_single_traversal(
        images, ages_years, "Age (years)",
        slice_idx=slice_idx, output_path=output_path
    )

    return fig


def visualize_ici_traversal(
    model: ConvVAE,
    image: torch.Tensor,
    n_steps: int = 5,
    ici_range: Tuple[float, float] = (-1.0, 2.0),
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize ICI traversal (dimension 1).

    Generates images showing how the cardiac structure changes
    as ICI treatment status varies (0=no, 1=yes).

    Args:
        model: Trained VAE model
        image: Input image to encode
        n_steps: Number of steps in traversal
        ici_range: Range for ICI dimension (extends beyond 0-1 to see extrapolation)
        slice_idx: Slice to visualize
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
    # Encode the input image
    z_base = encode_image(model, image)

    # Create ICI values
    ici_values = np.linspace(ici_range[0], ici_range[1], n_steps)

    # Traverse ICI dimension (dim 1)
    images = traverse_latent_dimension(model, z_base, dim=1, values=ici_values)

    # Visualize with labels
    labels = [f"ICI={v:.1f}" for v in ici_values]

    fig = visualize_single_traversal(
        images, ici_values, "ICI Treatment",
        slice_idx=slice_idx, output_path=output_path
    )

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
    """Visualize 2D grid of Age x ICI traversal.

    Creates a grid showing the combined effect of varying both
    age and ICI simultaneously.

    Args:
        model: Trained VAE model
        image: Input image to encode
        age_mean/std: Age normalization parameters
        age_steps: Number of age values
        ici_steps: Number of ICI values
        age_range: (min_age, max_age) in years
        ici_range: (min_ici, max_ici)
        slice_idx: Slice to visualize
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
    # Encode the input image
    z_base = encode_image(model, image)

    # Create value grids
    ages_years = np.linspace(age_range[0], age_range[1], age_steps)
    ages_normalized = (ages_years - age_mean) / age_std
    ici_values = np.linspace(ici_range[0], ici_range[1], ici_steps)

    # Create figure
    fig, axes = plt.subplots(ici_steps, age_steps, figsize=(2.5 * age_steps, 2.5 * ici_steps))

    if ici_steps == 1:
        axes = axes.reshape(1, -1)
    if age_steps == 1:
        axes = axes.reshape(-1, 1)

    for i, ici_val in enumerate(ici_values):
        for j, (age_years, age_norm) in enumerate(zip(ages_years, ages_normalized)):
            # Modify both dimensions
            z_modified = z_base.clone()
            z_modified[0, 0] = age_norm  # Age
            z_modified[0, 1] = ici_val   # ICI

            # Decode
            with torch.no_grad():
                recon = model.decode(z_modified)

            # Extract slice - slice along D dimension to get (H, W)
            img_np = recon[0, 0, slice_idx, :, :].cpu().numpy()

            # Plot
            ax = axes[i, j]
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')

            # Labels
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


def visualize_reconstruction_comparison(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    n_samples: int = 4,
    slice_idx: int = 40,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare original images with their reconstructions.

    Args:
        model: Trained VAE model
        dataset: Dataset to sample from
        n_samples: Number of samples to show
        slice_idx: Slice to visualize
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
    model.eval()

    # Sample random indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i, idx in enumerate(indices):
        image, label, attributes, subject_id = dataset[idx]

        # Original - slice along D dimension to get (H, W)
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


def visualize_latent_distribution(
    model: ConvVAE,
    dataset: AttriVAEDataset,
    dims: List[int] = [0, 1],
    max_samples: int = 100,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize the distribution of latent dimensions.

    Shows how age (dim 0) and ICI (dim 1) are distributed in latent space,
    colored by their actual attribute values.

    Args:
        model: Trained VAE model
        dataset: Dataset to analyze
        dims: Which dimensions to visualize
        max_samples: Maximum samples to use
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
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
        all_ages.append(attributes[0].item())  # Normalized age
        all_ici.append(attributes[1].item())    # ICI
        all_labels.append(label)

    all_z = np.concatenate(all_z, axis=0)
    all_ages = np.array(all_ages)
    all_ici = np.array(all_ici)
    all_labels = np.array(all_labels)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Dim 0 (Age) vs actual age
    ax = axes[0, 0]
    scatter = ax.scatter(all_z[:, 0], all_ages, c=all_labels, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Latent Dim 0 (Age)', fontsize=11)
    ax.set_ylabel('Actual Age (normalized)', fontsize=11)
    ax.set_title('Age Disentanglement', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='CHIP Label')

    # Add correlation
    corr = np.corrcoef(all_z[:, 0], all_ages)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    # Plot 2: Dim 1 (ICI) vs actual ICI
    ax = axes[0, 1]
    # Jitter ICI for better visualization
    ici_jittered = all_ici + np.random.normal(0, 0.05, len(all_ici))
    scatter = ax.scatter(all_z[:, 1], ici_jittered, c=all_labels, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Latent Dim 1 (ICI)', fontsize=11)
    ax.set_ylabel('Actual ICI (jittered)', fontsize=11)
    ax.set_title('ICI Disentanglement', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='CHIP Label')

    # Add correlation
    corr = np.corrcoef(all_z[:, 1], all_ici)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    # Plot 3: Dim 0 vs Dim 1 colored by age
    ax = axes[1, 0]
    scatter = ax.scatter(all_z[:, 0], all_z[:, 1], c=all_ages, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Latent Dim 0 (Age)', fontsize=11)
    ax.set_ylabel('Latent Dim 1 (ICI)', fontsize=11)
    ax.set_title('Latent Space (colored by Age)', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Age (norm)')

    # Plot 4: Dim 0 vs Dim 1 colored by ICI
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


def run_full_visualization(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    n_samples: int = 4,
    slice_idx: int = 40,
):
    """Run full visualization suite.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to config YAML
        output_dir: Directory to save visualizations
        n_samples: Number of samples for some visualizations
        slice_idx: Slice index to visualize
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    data_config = config.get('data', {})

    # Load model
    model = load_model(checkpoint_path, model_config)

    # Load dataset
    print("[Data] Loading dataset...")
    data = np.load(data_config['data_file'], allow_pickle=True).tolist()
    extractor = create_patient_extractor(data_config['spreadsheet'])
    age_stats = compute_age_stats(data, extractor)

    dataset = AttriVAEDataset(
        data, extractor,
        target_size=tuple(model_config.get('target_size', [80, 80, 80])),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False,
    )
    print(f"[Data] Loaded {len(dataset)} samples")

    # Select sample images for traversal
    sample_indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    # 1. Reconstruction comparison
    print("\n[Viz] Generating reconstruction comparison...")
    visualize_reconstruction_comparison(
        model, dataset, n_samples=n_samples, slice_idx=slice_idx,
        output_path=str(output_dir / "reconstruction_comparison.png")
    )

    # 2. Latent distribution
    print("[Viz] Generating latent distribution plot...")
    visualize_latent_distribution(
        model, dataset, max_samples=100,
        output_path=str(output_dir / "latent_distribution.png")
    )

    # 3. Age traversal for each sample
    print("[Viz] Generating age traversals...")
    for i, idx in enumerate(sample_indices):
        image, label, attrs, subj_id = dataset[idx]
        raw = dataset.get_raw_attributes(idx)

        visualize_age_traversal(
            model, image,
            age_mean=age_stats['mean'],
            age_std=age_stats['std'],
            n_steps=9,
            slice_idx=slice_idx,
            output_path=str(output_dir / f"age_traversal_sample{i}_{subj_id}.png")
        )

    # 4. ICI traversal for each sample
    print("[Viz] Generating ICI traversals...")
    for i, idx in enumerate(sample_indices):
        image, label, attrs, subj_id = dataset[idx]

        visualize_ici_traversal(
            model, image,
            n_steps=5,
            slice_idx=slice_idx,
            output_path=str(output_dir / f"ici_traversal_sample{i}_{subj_id}.png")
        )

    # 5. 2D Age x ICI grid for first sample
    print("[Viz] Generating 2D Age x ICI grid...")
    image, _, _, subj_id = dataset[sample_indices[0]]
    visualize_2d_grid(
        model, image,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        age_steps=5,
        ici_steps=3,
        slice_idx=slice_idx,
        output_path=str(output_dir / f"age_ici_grid_{subj_id}.png")
    )

    print(f"\n[Done] All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AttriVAE latent space manipulation (attribute scanning)"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(Path(__file__).parent / 'configs' / 'train_attrivae.yaml'),
        help='Path to config YAML'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='Number of samples for traversal visualizations'
    )
    parser.add_argument(
        '--slice_idx',
        type=int,
        default=40,
        help='Slice index to visualize (0-79)'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(__file__).parent / 'outputs' / 'visualizations' / timestamp)

    run_full_visualization(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        slice_idx=args.slice_idx,
    )


if __name__ == "__main__":
    main()
