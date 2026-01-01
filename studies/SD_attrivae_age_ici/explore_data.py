#!/usr/bin/env python3
"""
Data Exploration Script for AttriVAE Age/ICI Study

This script explores the organized LGE cardiac MRI data to verify:
1. Data structure and available fields
2. Short-axis (sas) images exist and have expected format
3. CHIP labels are present
4. Age/ICI can be loaded from spreadsheet
5. Statistics and sample visualizations

Usage:
    python explore_data.py [--save-figures]

Author: Claude
Date: 2025-12-28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LGE_CHIP_ROOT = Path("/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LGE_CHIP_ROOT) not in sys.path:
    sys.path.insert(0, str(LGE_CHIP_ROOT))

# Data paths
ORGANIZED_DATA_FILE = LGE_CHIP_ROOT / "data" / "2025-06-01-onc-cohort-144-with-serial-scans-and-103-LGE-masks.npy"
SPREADSHEET_PATH = LGE_CHIP_ROOT / "link_project_data" / "CHIP ICI MI CM outcomes - Updated 2025-10-09.xlsx"

# Output directory for figures
OUTPUT_DIR = Path(__file__).parent / "outputs" / "data_exploration"


def load_organized_data(data_file: Path) -> List[Dict[str, Any]]:
    """Load the organized NPY data file."""
    print(f"\n{'='*60}")
    print("Loading Organized Data")
    print(f"{'='*60}")
    print(f"File: {data_file}")

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = np.load(data_file, allow_pickle=True).tolist()
    print(f"Loaded {len(data)} patient records")
    return data


def explore_data_structure(data: List[Dict[str, Any]]) -> None:
    """Explore the structure of the organized data."""
    print(f"\n{'='*60}")
    print("Data Structure Exploration")
    print(f"{'='*60}")

    if not data:
        print("WARNING: Data is empty!")
        return

    # Check first patient's structure
    sample = data[0]
    print(f"\nFirst patient keys:")
    for key in sorted(sample.keys()):
        value = sample[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            if value and isinstance(value[0], np.ndarray):
                print(f"  {key}: list of {len(value)} ndarrays, first shape={value[0].shape}")
            else:
                print(f"  {key}: list of {len(value)} items")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")


def check_sas_images(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check short-axis (sas) images availability and format."""
    print(f"\n{'='*60}")
    print("Short-Axis (SAS) Images Check")
    print(f"{'='*60}")

    # Check for sas_images or sas_volume or sas_segmented_* (raw format)
    sas_key = None
    for key in ['sas_images', 'sas_volume', 'sas_segmented_PSIR_images', 'sas_single_shot_PSIR_images']:
        if key in data[0]:
            sas_key = key
            break

    if sas_key is None:
        print("ERROR: No SAS images found!")
        print(f"Available keys: {list(data[0].keys())}")
        return {'sas_key': None, 'stats': None}

    print(f"Found SAS data with key: '{sas_key}'")

    # Collect statistics
    slice_counts = []
    shapes = []
    has_sas = 0
    missing_sas = 0

    for patient in data:
        sas_data = patient.get(sas_key)

        if sas_key in ['sas_images', 'sas_segmented_PSIR_images', 'sas_segmented_MAG_images',
                       'sas_single_shot_PSIR_images', 'sas_single_shot_MAG_images']:
            # List of slices (each slice is a dict with 'image' key or numpy array)
            if sas_data and len(sas_data) > 0:
                has_sas += 1
                slice_counts.append(len(sas_data))
                # Check if slices are dicts or arrays
                first_slice = sas_data[0]
                if isinstance(first_slice, dict):
                    if 'image' in first_slice:
                        shapes.append(first_slice['image'].shape)
                    else:
                        shapes.append(('dict_no_image',))
                elif isinstance(first_slice, np.ndarray):
                    shapes.append(first_slice.shape)
                else:
                    shapes.append((type(first_slice).__name__,))
            else:
                missing_sas += 1
                slice_counts.append(0)
        elif sas_key == 'sas_volume':
            # Volume (C, D, H, W)
            if sas_data is not None and isinstance(sas_data, np.ndarray) and sas_data.size > 0:
                has_sas += 1
                slice_counts.append(sas_data.shape[1] if len(sas_data.shape) >= 2 else 1)
                shapes.append(sas_data.shape)
            else:
                missing_sas += 1
                slice_counts.append(0)
        else:
            # Unknown format
            if sas_data:
                has_sas += 1
                slice_counts.append(len(sas_data) if hasattr(sas_data, '__len__') else 1)
            else:
                missing_sas += 1
                slice_counts.append(0)

    print(f"\nPatients with SAS data: {has_sas}/{len(data)}")
    print(f"Patients missing SAS data: {missing_sas}/{len(data)}")

    if slice_counts:
        print(f"\nSlice count statistics:")
        print(f"  Min: {min(slice_counts)}")
        print(f"  Max: {max(slice_counts)}")
        print(f"  Mean: {np.mean(slice_counts):.1f}")
        print(f"  Median: {np.median(slice_counts):.0f}")

    if shapes:
        unique_shapes = list(set(shapes))
        print(f"\nUnique slice shapes: {unique_shapes[:5]}...")

    # Check channel info
    if 'sas_channel_info' in data[0]:
        channel_info = data[0]['sas_channel_info']
        print(f"\nChannel info ({len(channel_info)} channels):")
        for i, info in enumerate(channel_info[:5]):
            print(f"  [{i}] {info}")
        if len(channel_info) > 5:
            print(f"  ... and {len(channel_info)-5} more channels")

    return {
        'sas_key': sas_key,
        'has_sas': has_sas,
        'missing_sas': missing_sas,
        'slice_counts': slice_counts,
        'shapes': shapes
    }


def check_chip_labels(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check CHIP labels availability."""
    print(f"\n{'='*60}")
    print("CHIP Label Check")
    print(f"{'='*60}")

    # Look for CHIP label key
    label_key = None
    for key in ['CHIP_label', 'chip_label', 'label']:
        if key in data[0]:
            label_key = key
            break

    if label_key is None:
        print("ERROR: No CHIP label found!")
        print(f"Available keys: {list(data[0].keys())}")
        return {'label_key': None}

    print(f"Found label with key: '{label_key}'")

    # Count labels
    labels = [patient.get(label_key, -1) for patient in data]
    unique_labels = sorted(set(labels))

    print(f"\nUnique labels: {unique_labels}")
    print(f"\nLabel distribution:")
    for label in unique_labels:
        count = labels.count(label)
        print(f"  Label {label}: {count} ({100*count/len(labels):.1f}%)")

    return {
        'label_key': label_key,
        'labels': labels,
        'distribution': {label: labels.count(label) for label in unique_labels}
    }


def load_patient_metadata(data: List[Dict[str, Any]], spreadsheet_path: Path) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """Load age and ICI from spreadsheet using PatientDataExtractor."""
    print(f"\n{'='*60}")
    print("Patient Metadata (Age/ICI) Check")
    print(f"{'='*60}")

    print(f"Spreadsheet: {spreadsheet_path}")

    if not spreadsheet_path.exists():
        print("ERROR: Spreadsheet not found!")
        return {}, {}

    try:
        from module.data.siteIO.patient_data_extractor import PatientDataExtractor

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

        extractor = PatientDataExtractor(
            excel_file=str(spreadsheet_path),
            columns_config=columns_config,
            key_column='Anonymized Code'
        )

        print(f"Loaded extractor with {len(extractor)} records")

    except Exception as e:
        print(f"ERROR loading PatientDataExtractor: {e}")
        print("Falling back to direct pandas read...")

        import pandas as pd
        df = pd.read_excel(spreadsheet_path)
        print(f"Spreadsheet columns: {list(df.columns)}")
        return {}, {'error': str(e)}

    # Extract metadata for each patient
    metadata = {}
    ages = []
    icis = []
    missing_age = 0
    missing_ici = 0

    for patient in data:
        subject_id = patient.get('subject_id', '')
        patient_data = extractor.get_patient_data(subject_id)

        age = patient_data.get('age')
        ici = patient_data.get('ici')

        metadata[subject_id] = {'age': age, 'ici': ici}

        if age is not None:
            ages.append(age)
        else:
            missing_age += 1

        if ici is not None:
            icis.append(ici)
        else:
            missing_ici += 1

    print(f"\nAge statistics:")
    print(f"  Valid: {len(ages)}/{len(data)} ({100*len(ages)/len(data):.1f}%)")
    print(f"  Missing: {missing_age}/{len(data)}")
    if ages:
        print(f"  Min: {min(ages)}")
        print(f"  Max: {max(ages)}")
        print(f"  Mean: {np.mean(ages):.1f}")
        print(f"  Std: {np.std(ages):.1f}")

    print(f"\nICI statistics:")
    print(f"  Valid: {len(icis)}/{len(data)} ({100*len(icis)/len(data):.1f}%)")
    print(f"  Missing: {missing_ici}/{len(data)}")
    if icis:
        ici_counts = {0: icis.count(0), 1: icis.count(1)}
        print(f"  ICI=0 (No): {ici_counts[0]} ({100*ici_counts[0]/len(icis):.1f}%)")
        print(f"  ICI=1 (Yes/Trial): {ici_counts[1]} ({100*ici_counts[1]/len(icis):.1f}%)")

    stats = {
        'ages': ages,
        'icis': icis,
        'missing_age': missing_age,
        'missing_ici': missing_ici
    }

    return metadata, stats


def get_middle_slices(slices: List[np.ndarray], n: int = 3) -> List[np.ndarray]:
    """Get middle n slices from a list of slices."""
    if not slices:
        return []

    total = len(slices)
    if total <= n:
        return slices

    start = (total - n) // 2
    return slices[start:start + n]


def visualize_samples(data: List[Dict[str, Any]], sas_key: str,
                     metadata: Dict, label_key: str,
                     n_samples: int = 6, save_path: Optional[Path] = None) -> None:
    """Visualize sample images from the dataset."""
    print(f"\n{'='*60}")
    print("Sample Visualization")
    print(f"{'='*60}")

    # Select samples with valid data
    valid_samples = []
    for patient in data:
        sas_data = patient.get(sas_key)
        if sas_data and len(sas_data) > 0:
            # Check if slices have images
            first_slice = sas_data[0]
            if isinstance(first_slice, dict) and 'image' in first_slice:
                valid_samples.append(patient)
            elif isinstance(first_slice, np.ndarray):
                valid_samples.append(patient)

    print(f"Found {len(valid_samples)} patients with valid SAS data")

    if not valid_samples:
        print("No valid samples to visualize!")
        return

    # Sample patients
    np.random.seed(42)
    sample_indices = np.random.choice(len(valid_samples), min(n_samples, len(valid_samples)), replace=False)
    samples = [valid_samples[i] for i in sample_indices]

    # Create figure
    n_cols = 3  # Middle 3 slices
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, patient in enumerate(samples):
        subject_id = patient.get('subject_id', 'Unknown')
        label = patient.get(label_key, -1)
        meta = metadata.get(subject_id, {})
        age = meta.get('age', 'N/A')
        ici = meta.get('ici', 'N/A')

        # Get SAS data
        sas_data = patient[sas_key]

        # Handle different formats
        if isinstance(sas_data[0], dict):
            # List of dicts with 'image' key
            slices = [s['image'] for s in sas_data if 'image' in s]
        elif isinstance(sas_data[0], np.ndarray):
            # List of numpy arrays
            slices = sas_data
        else:
            slices = []

        middle_slices = get_middle_slices(slices, n=3)

        for col in range(n_cols):
            ax = axes[row, col]

            if col < len(middle_slices):
                img = middle_slices[col]
                # If (C, H, W), take first channel
                if len(img.shape) == 3:
                    img = img[0]

                ax.imshow(img, cmap='gray')
                ax.axis('off')

                if col == 0:
                    title = f"ID: {subject_id[:8]}...\nCHIP={label}, Age={age}, ICI={ici}"
                    ax.set_title(title, fontsize=9)
                else:
                    ax.set_title(f"Slice {col+1}", fontsize=9)
            else:
                ax.axis('off')
                ax.set_title("N/A", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()


def summarize_for_attrivae(data: List[Dict], sas_stats: Dict, label_stats: Dict,
                           metadata: Dict, meta_stats: Dict) -> None:
    """Summarize data readiness for AttriVAE training."""
    print(f"\n{'='*60}")
    print("Summary: Data Readiness for AttriVAE")
    print(f"{'='*60}")

    total = len(data)
    sas_key = sas_stats.get('sas_key')

    # Count valid patients (have SAS + CHIP + age + ICI)
    valid_count = 0
    has_sas_count = 0
    has_label_count = 0
    has_age_count = 0
    has_ici_count = 0

    for patient in data:
        subject_id = patient.get('subject_id', '')

        # Check SAS
        has_sas = False
        if sas_key:
            sas_data = patient.get(sas_key)
            if sas_data and len(sas_data) > 0:
                first_slice = sas_data[0]
                if isinstance(first_slice, dict) and 'image' in first_slice:
                    has_sas = True
                elif isinstance(first_slice, np.ndarray):
                    has_sas = True

        if has_sas:
            has_sas_count += 1

        # Check label
        label_key = label_stats.get('label_key')
        has_label = label_key and patient.get(label_key) is not None
        if has_label:
            has_label_count += 1

        # Check metadata
        meta = metadata.get(subject_id, {})
        has_age = meta.get('age') is not None
        has_ici = meta.get('ici') is not None

        if has_age:
            has_age_count += 1
        if has_ici:
            has_ici_count += 1

        if has_sas and has_label and has_age and has_ici:
            valid_count += 1

    print(f"\nTotal patients: {total}")
    print(f"Valid for training: {valid_count} ({100*valid_count/total:.1f}%)")
    print(f"  - Have SAS images: {has_sas_count}")
    print(f"  - Have CHIP label: {has_label_count}")
    print(f"  - Have age: {has_age_count}")
    print(f"  - Have ICI: {has_ici_count}")

    # Recommendation
    print(f"\nRecommendation:")
    if valid_count >= 50:
        print(f"  ✓ Sufficient data for training ({valid_count} patients)")
    else:
        print(f"  ⚠ Limited data ({valid_count} patients) - consider augmentation")

    if sas_stats.get('slice_counts'):
        median_slices = np.median(sas_stats['slice_counts'])
        if median_slices >= 3:
            print(f"  ✓ Sufficient slices for middle-3 selection (median={median_slices:.0f})")
        else:
            print(f"  ⚠ Few slices (median={median_slices:.0f}) - some patients may have <3 slices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-figures', action='store_true', help='Save figures to file')
    parser.add_argument('--data-file', type=str, default=None, help='Override data file path')
    args = parser.parse_args()

    # Load data
    data_file = Path(args.data_file) if args.data_file else ORGANIZED_DATA_FILE
    data = load_organized_data(data_file)

    # Explore structure
    explore_data_structure(data)

    # Check SAS images
    sas_stats = check_sas_images(data)

    # Check CHIP labels
    label_stats = check_chip_labels(data)

    # Load patient metadata
    metadata, meta_stats = load_patient_metadata(data, SPREADSHEET_PATH)

    # Visualize samples
    if sas_stats.get('sas_key') and label_stats.get('label_key'):
        save_path = OUTPUT_DIR / "sample_images.png" if args.save_figures else None
        visualize_samples(
            data,
            sas_stats['sas_key'],
            metadata,
            label_stats['label_key'],
            save_path=save_path
        )

    # Summary
    summarize_for_attrivae(data, sas_stats, label_stats, metadata, meta_stats)

    print(f"\n{'='*60}")
    print("Exploration Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
