"""
Dataset class for AttriVAE with Age/ICI attributes.

This module provides a PyTorch Dataset for training AttriVAE on LGE cardiac MRI
with age and ICI treatment as disentangled attributes.

Data format:
- Input: Short-axis PSIR images (middle 3 slices) → 3D volume (1, 80, 80, 80)
- Label: CHIP status (binary 0/1)
- Attributes: [normalized_age, ici] → tensor shape (2,)

Author: Claude
Date: 2025-12-28
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

# Add path for PatientDataExtractor
LGE_CHIP_ROOT = Path("/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2")
if str(LGE_CHIP_ROOT) not in sys.path:
    sys.path.insert(0, str(LGE_CHIP_ROOT))

from module.data.siteIO.patient_data_extractor import PatientDataExtractor


def get_middle_slices(slices: List[np.ndarray], n: int = 3) -> List[np.ndarray]:
    """Get middle n slices from a list of slices.

    Args:
        slices: List of 2D numpy arrays
        n: Number of middle slices to extract

    Returns:
        List of middle n slices. If fewer than n slices, returns all with padding.
    """
    if not slices:
        return []

    total = len(slices)
    if total <= n:
        # Pad with edge slices if not enough
        result = slices.copy()
        while len(result) < n:
            if len(result) % 2 == 0:
                result.insert(0, result[0])  # Pad at beginning
            else:
                result.append(result[-1])  # Pad at end
        return result

    start = (total - n) // 2
    return slices[start:start + n]


def resize_3d_volume(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Resize a 3D volume to target size.

    Args:
        volume: 3D numpy array (D, H, W)
        target_size: Target size (D, H, W)

    Returns:
        Resized volume
    """
    return resize(volume, target_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)


class AttriVAEDataset(Dataset):
    """Dataset for AttriVAE training with age/ICI attributes.

    This dataset:
    1. Loads short-axis PSIR images from organized NPY data
    2. Extracts middle 3 slices and stacks to 3D volume
    3. Resizes to target size (default: 80×80×80)
    4. Loads age/ICI from patient metadata spreadsheet
    5. Normalizes age using z-score, keeps ICI binary

    Args:
        data: List of patient dictionaries from organized NPY
        patient_extractor: PatientDataExtractor for age/ICI lookup
        target_size: Target volume size (D, H, W), default (80, 80, 80)
        sas_key: Key for SAS images in patient dict
        age_mean: Mean age for z-score normalization
        age_std: Std age for z-score normalization
        augment: Whether to apply data augmentation

    Returns per __getitem__:
        Tuple of (image, label, attributes, subject_id):
        - image: torch.Tensor shape (1, D, H, W)
        - label: int (CHIP label 0/1)
        - attributes: torch.Tensor shape (2,) [normalized_age, ici]
        - subject_id: str
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        patient_extractor: PatientDataExtractor,
        target_size: Tuple[int, int, int] = (80, 80, 80),
        sas_key: str = 'sas_segmented_PSIR_images',
        age_mean: float = 63.7,
        age_std: float = 14.0,
        augment: bool = False,
    ):
        self.patient_extractor = patient_extractor
        self.target_size = target_size
        self.sas_key = sas_key
        self.age_mean = age_mean
        self.age_std = age_std
        self.augment = augment

        # Filter to valid patients (have SAS images, CHIP label, age, ICI)
        self.valid_data = []
        self.skipped_reasons = {'no_sas': 0, 'no_label': 0, 'no_age': 0, 'no_ici': 0}

        for patient in data:
            subject_id = patient.get('subject_id', '')

            # Check SAS images
            sas_data = patient.get(sas_key)
            if not sas_data or len(sas_data) == 0:
                self.skipped_reasons['no_sas'] += 1
                continue

            # Check slices have 'image' key
            first_slice = sas_data[0]
            if isinstance(first_slice, dict) and 'image' not in first_slice:
                self.skipped_reasons['no_sas'] += 1
                continue

            # Check CHIP label
            if patient.get('CHIP_label') is None:
                self.skipped_reasons['no_label'] += 1
                continue

            # Check age/ICI from spreadsheet
            meta = patient_extractor.get_patient_data(subject_id)
            if meta.get('age') is None:
                self.skipped_reasons['no_age'] += 1
                continue
            if meta.get('ici') is None:
                self.skipped_reasons['no_ici'] += 1
                continue

            self.valid_data.append(patient)

        print(f"[AttriVAEDataset] Valid: {len(self.valid_data)}/{len(data)}")
        print(f"  Skipped: {self.skipped_reasons}")

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
        patient = self.valid_data[idx]
        subject_id = patient['subject_id']

        # Get SAS slices
        sas_data = patient[self.sas_key]

        # Extract images from slice dicts
        if isinstance(sas_data[0], dict):
            slices = [s['image'] for s in sas_data if 'image' in s]
        else:
            slices = list(sas_data)

        # Get middle 3 slices
        middle_slices = get_middle_slices(slices, n=3)

        # Resize each slice to target H, W (slices may have different shapes)
        target_hw = (self.target_size[1], self.target_size[2])  # (H, W)
        resized_slices = []
        for s in middle_slices:
            if s.shape != target_hw:
                s_resized = resize(s, target_hw, order=1, preserve_range=True, anti_aliasing=True)
            else:
                s_resized = s
            resized_slices.append(s_resized.astype(np.float32))

        # Stack to 3D volume (D, H, W)
        volume = np.stack(resized_slices, axis=0)

        # Normalize to [0, 1]
        volume = normalize_to_01(volume)

        # Resize depth if needed (e.g., 3 slices -> target_size[0])
        if volume.shape[0] != self.target_size[0]:
            volume = resize_3d_volume(volume, self.target_size)

        # Add channel dimension (1, D, H, W)
        volume = volume[np.newaxis, ...]

        # Apply augmentation if enabled
        if self.augment:
            volume = self._apply_augmentation(volume)

        # Convert to tensor
        image = torch.from_numpy(volume).float()

        # Get label
        label = int(patient['CHIP_label'])

        # Get attributes (age, ICI)
        meta = self.patient_extractor.get_patient_data(subject_id)
        age_raw = float(meta['age'])
        ici = float(meta['ici'])

        # Normalize age
        age_normalized = (age_raw - self.age_mean) / (self.age_std + 1e-8)

        # Create attributes tensor [normalized_age, ici]
        attributes = torch.tensor([age_normalized, ici], dtype=torch.float32)

        return image, label, attributes, subject_id

    def _apply_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Apply random augmentations to volume.

        Args:
            volume: 4D array (C, D, H, W)

        Returns:
            Augmented volume
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=3).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()

        # Random 90-degree rotation in xy plane
        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k=k, axes=(2, 3)).copy()

        return volume

    def get_raw_attributes(self, idx: int) -> Dict[str, Any]:
        """Get raw (non-normalized) attributes for a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with raw age and ICI values
        """
        patient = self.valid_data[idx]
        subject_id = patient['subject_id']
        meta = self.patient_extractor.get_patient_data(subject_id)
        return {
            'age': meta.get('age'),
            'ici': meta.get('ici'),
            'chip_label': patient.get('CHIP_label')
        }


def compute_age_stats(
    data: List[Dict[str, Any]],
    patient_extractor: PatientDataExtractor,
    sas_key: str = 'sas_segmented_PSIR_images'
) -> Dict[str, float]:
    """Compute age statistics from valid patients.

    Args:
        data: List of patient dictionaries
        patient_extractor: PatientDataExtractor for age lookup
        sas_key: Key for SAS images

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'n' for age
    """
    ages = []

    for patient in data:
        subject_id = patient.get('subject_id', '')

        # Check has valid SAS
        sas_data = patient.get(sas_key)
        if not sas_data or len(sas_data) == 0:
            continue

        # Get age
        meta = patient_extractor.get_patient_data(subject_id)
        age = meta.get('age')
        if age is not None:
            ages.append(float(age))

    if not ages:
        return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0, 'n': 0}

    return {
        'mean': float(np.mean(ages)),
        'std': float(np.std(ages)),
        'min': float(np.min(ages)),
        'max': float(np.max(ages)),
        'n': len(ages)
    }


def create_patient_extractor(spreadsheet_path: str) -> PatientDataExtractor:
    """Create a PatientDataExtractor with age/ICI columns configured.

    Args:
        spreadsheet_path: Path to Excel spreadsheet

    Returns:
        Configured PatientDataExtractor
    """
    columns_config = {
        'age': {
            'source_column': 'age',
            'type': 'int',
            'default': None
        },
        'ici': {
            'source_column': 'ICI Y (1) or N (0) trial (2)',
            'type': 'category',
            'mapping': {
                '0': 0, '0.0': 0,
                '1': 1, '1.0': 1,
                '2': 1, '2.0': 1  # Trial mapped to 1
            },
            'default': None
        }
    }

    return PatientDataExtractor(
        excel_file=spreadsheet_path,
        columns_config=columns_config,
        key_column='Anonymized Code'
    )


def create_stratified_split(
    data: List[Dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 42,
    label_key: str = 'CHIP_label'
) -> Tuple[List[Dict], List[Dict]]:
    """Create stratified train/val split.

    Args:
        data: List of patient dictionaries
        val_fraction: Fraction for validation
        seed: Random seed
        label_key: Key for stratification labels

    Returns:
        Tuple of (train_data, val_data)
    """
    np.random.seed(seed)

    # Group by label
    class_0 = [d for d in data if d.get(label_key) == 0]
    class_1 = [d for d in data if d.get(label_key) == 1]

    # Shuffle
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    # Split each class
    n_val_0 = max(1, int(len(class_0) * val_fraction))
    n_val_1 = max(1, int(len(class_1) * val_fraction))

    val_data = class_0[:n_val_0] + class_1[:n_val_1]
    train_data = class_0[n_val_0:] + class_1[n_val_1:]

    # Shuffle again
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    print(f"[Split] Train: {len(train_data)} (class0: {len(class_0)-n_val_0}, class1: {len(class_1)-n_val_1})")
    print(f"[Split] Val: {len(val_data)} (class0: {n_val_0}, class1: {n_val_1})")

    return train_data, val_data


# Testing code
if __name__ == "__main__":
    import sys

    # Paths
    DATA_FILE = LGE_CHIP_ROOT / "data" / "2025-06-01-onc-cohort-144-with-serial-scans-and-103-LGE-masks.npy"
    SPREADSHEET = LGE_CHIP_ROOT / "link_project_data" / "CHIP ICI MI CM outcomes - Updated 2025-10-09.xlsx"

    print("Loading data...")
    data = np.load(DATA_FILE, allow_pickle=True).tolist()
    print(f"Loaded {len(data)} patients")

    print("\nCreating patient extractor...")
    extractor = create_patient_extractor(str(SPREADSHEET))

    print("\nComputing age stats...")
    age_stats = compute_age_stats(data, extractor)
    print(f"Age stats: {age_stats}")

    print("\nCreating train/val split...")
    train_data, val_data = create_stratified_split(data)

    print("\nCreating datasets...")
    train_ds = AttriVAEDataset(
        train_data, extractor,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=True
    )
    val_ds = AttriVAEDataset(
        val_data, extractor,
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False
    )

    print(f"\nTrain dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")

    # Test a sample
    print("\nTesting sample retrieval...")
    image, label, attributes, subject_id = train_ds[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label}")
    print(f"  Attributes: {attributes} (normalized_age, ici)")
    print(f"  Subject ID: {subject_id}")

    raw = train_ds.get_raw_attributes(0)
    print(f"  Raw attributes: {raw}")

    print("\nDataset test complete!")
