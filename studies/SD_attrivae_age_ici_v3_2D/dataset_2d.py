"""
2D Dataset class for AttriVAE with Age/ICI attributes.

This module provides a PyTorch Dataset that treats each cardiac MRI slice
as an independent 2D sample, rather than stacking slices into 3D volumes.

Key differences from the 3D version:
- Each patient contributes 3 samples (middle 3 slices)
- Training samples increase ~3x (e.g., 94 patients → ~282 slices)
- Slices share the same patient attributes (age, ICI, CHIP label)
- Input shape: (1, 80, 80) instead of (1, 80, 80, 80)

Data format:
- Input: Single 2D slice (1, 80, 80)
- Label: CHIP status (binary 0/1)
- Attributes: [normalized_age, ici] → tensor shape (2,)

Author: Claude
Date: 2025-12-31
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


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)


class AttriVAEDataset2D(Dataset):
    """2D Dataset for AttriVAE training with age/ICI attributes.

    This dataset treats each cardiac MRI slice as an independent sample,
    effectively increasing the dataset size by 3x (3 slices per patient).

    Key features:
    - Each __getitem__ returns a single 2D slice
    - Patient attributes (age, ICI, CHIP) are shared across all slices from same patient
    - Augmentation applied independently to each slice

    Args:
        data: List of patient dictionaries from organized NPY
        patient_extractor: PatientDataExtractor for age/ICI lookup
        target_size: Target slice size (H, W), default (80, 80)
        sas_key: Key for SAS images in patient dict
        age_mean: Mean age for z-score normalization
        age_std: Std age for z-score normalization
        augment: Whether to apply data augmentation
        n_slices: Number of middle slices to extract per patient (default 3)

    Returns per __getitem__:
        Tuple of (image, label, attributes, subject_id, slice_idx):
        - image: torch.Tensor shape (1, H, W)
        - label: int (CHIP label 0/1)
        - attributes: torch.Tensor shape (2,) [normalized_age, ici]
        - subject_id: str
        - slice_idx: int (0 to n_slices-1)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        patient_extractor: PatientDataExtractor,
        target_size: Tuple[int, int] = (80, 80),
        sas_key: str = 'sas_segmented_PSIR_images',
        age_mean: float = 63.7,
        age_std: float = 14.0,
        augment: bool = False,
        n_slices: int = 3,
    ):
        self.patient_extractor = patient_extractor
        self.target_size = target_size
        self.sas_key = sas_key
        self.age_mean = age_mean
        self.age_std = age_std
        self.augment = augment
        self.n_slices = n_slices

        # Build flat list of (patient_data, slice_idx) pairs
        self.samples = []
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

            # Extract slices
            if isinstance(sas_data[0], dict):
                slices = [s['image'] for s in sas_data if 'image' in s]
            else:
                slices = list(sas_data)

            middle_slices = get_middle_slices(slices, n=n_slices)

            # Add one sample per slice
            for slice_idx, slice_img in enumerate(middle_slices):
                self.samples.append({
                    'patient': patient,
                    'slice_img': slice_img,
                    'slice_idx': slice_idx,
                    'subject_id': subject_id,
                })

        n_patients = len(set(s['subject_id'] for s in self.samples))
        print(f"[AttriVAEDataset2D] Valid: {n_patients} patients, {len(self.samples)} slices")
        print(f"  Skipped: {self.skipped_reasons}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, str, int]:
        sample = self.samples[idx]
        patient = sample['patient']
        slice_img = sample['slice_img']
        slice_idx = sample['slice_idx']
        subject_id = sample['subject_id']

        # Resize slice to target size if needed
        if slice_img.shape != self.target_size:
            slice_img = resize(
                slice_img,
                self.target_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float32)

        # Normalize to [0, 1]
        slice_img = normalize_to_01(slice_img)

        # Add channel dimension (1, H, W)
        image = slice_img[np.newaxis, ...]

        # Apply augmentation if enabled
        if self.augment:
            image = self._apply_augmentation(image)

        # Convert to tensor
        image = torch.from_numpy(image.copy()).float()

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

        return image, label, attributes, subject_id, slice_idx

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to 2D image.

        Args:
            image: 3D array (C, H, W)

        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()

        return image

    def get_raw_attributes(self, idx: int) -> Dict[str, Any]:
        """Get raw (non-normalized) attributes for a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with raw age, ICI, slice_idx, and subject_id
        """
        sample = self.samples[idx]
        patient = sample['patient']
        subject_id = sample['subject_id']
        meta = self.patient_extractor.get_patient_data(subject_id)
        return {
            'age': meta.get('age'),
            'ici': meta.get('ici'),
            'chip_label': patient.get('CHIP_label'),
            'slice_idx': sample['slice_idx'],
            'subject_id': subject_id,
        }

    def get_patient_indices(self, subject_id: str) -> List[int]:
        """Get all sample indices for a given patient.

        Args:
            subject_id: Patient ID

        Returns:
            List of sample indices belonging to this patient
        """
        return [i for i, s in enumerate(self.samples) if s['subject_id'] == subject_id]

    def get_unique_patients(self) -> List[str]:
        """Get list of unique patient IDs in the dataset."""
        return list(set(s['subject_id'] for s in self.samples))


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
    """Create stratified train/val split at patient level.

    Important: Split is done at the patient level, not slice level,
    to prevent data leakage (slices from same patient should not appear
    in both train and val).

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

    print(f"[Split] Train: {len(train_data)} patients (class0: {len(class_0)-n_val_0}, class1: {len(class_1)-n_val_1})")
    print(f"[Split] Val: {len(val_data)} patients (class0: {n_val_0}, class1: {n_val_1})")

    return train_data, val_data


# Testing code
if __name__ == "__main__":
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

    print("\nCreating 2D datasets...")
    train_ds = AttriVAEDataset2D(
        train_data, extractor,
        target_size=(80, 80),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=True
    )
    val_ds = AttriVAEDataset2D(
        val_data, extractor,
        target_size=(80, 80),
        age_mean=age_stats['mean'],
        age_std=age_stats['std'],
        augment=False
    )

    print(f"\nTrain dataset: {len(train_ds)} slices from {len(train_ds.get_unique_patients())} patients")
    print(f"Val dataset: {len(val_ds)} slices from {len(val_ds.get_unique_patients())} patients")

    # Test a sample
    print("\nTesting sample retrieval...")
    image, label, attributes, subject_id, slice_idx = train_ds[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label}")
    print(f"  Attributes: {attributes} (normalized_age, ici)")
    print(f"  Subject ID: {subject_id}")
    print(f"  Slice index: {slice_idx}")

    raw = train_ds.get_raw_attributes(0)
    print(f"  Raw attributes: {raw}")

    # Test patient grouping
    print(f"\n  All indices for patient {subject_id}: {train_ds.get_patient_indices(subject_id)}")

    print("\n2D Dataset test complete!")
