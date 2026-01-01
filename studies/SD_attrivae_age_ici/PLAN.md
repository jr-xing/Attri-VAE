# AttriVAE Training Setup for LGE Cardiac MRI with Age/ICI Disentanglement

## Overview

Set up AttriVAE training on Yale LGE cardiac MRI dataset with two disentangled latent dimensions:
- **Dim 0**: Age (continuous integer)
- **Dim 1**: ICI treatment (binary 0/1)

Classification target: **CHIP label** (binary, from NPY data)

## Key Design Decisions

1. **Code location**: `/gpfs/gibbs/project/kwan/jx332/code/2025-12-Attri-VAE/studies/SD_attrivae_age_ici/`
2. **Input format**: Pseudo-3D volume from short-axis (sas) middle 3 slices, resized to 80×80×80
3. **Latent space**: 64 dimensions, first 2 constrained by attribute regularization (age, ICI)
4. **Data source**: Already-organized NPY at `/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2/data/2025-06-01-onc-cohort-144-with-serial-scans-and-103-LGE-masks.npy`
5. **Tabular data**: Load age/ICI from spreadsheet via PatientDataExtractor

## Files to Create/Modify

### 1. New Study Directory Structure (in AttriVAE repo)
```
/gpfs/gibbs/project/kwan/jx332/code/2025-12-Attri-VAE/
└── studies/
    └── SD_attrivae_age_ici/
        ├── configs/
        │   └── train_attrivae.yaml
        ├── dataset.py              # Custom dataset for AttriVAE
        ├── main.py                 # Entry point
        └── README.md
```

### 2. Model Modifications (model/model.py)

**Change**: Adapt ConvVAE for 2D-stacked input (1, D, 80, 80) where D = number of slices

The current model expects 80×80×80 3D volumes. Options:
- **Option A**: Keep 3D architecture, resize slices to create 80×80×80 volume (pad/interpolate depth)
- **Option B**: Modify to handle variable depth with adaptive pooling

**Recommended**: Option A - resize to 80×80×80 for simplicity

### 3. Custom Dataset (studies/SD_attrivae_age_ici/dataset.py)

```python
class AttriVAEDataset(torch.utils.data.Dataset):
    """Dataset for AttriVAE with age/ICI attributes."""

    def __init__(self, organized_data, patient_extractor, transforms,
                 view='sas', target_size=(80, 80, 80)):
        # Load organized data (already has sas_images as list of slices)
        # Extract short-axis slices and stack to 3D
        # Load age/ICI from patient_extractor
        # Normalize age (z-score), keep ICI binary

    def __getitem__(self, idx):
        # Returns: (image_3d, chip_label, attributes, subject_id)
        # image_3d: (1, 80, 80, 80) tensor
        # chip_label: 0 or 1
        # attributes: tensor([normalized_age, ici]) shape (2,)
```

**Key steps**:
1. Load from organized data (already processed by organize_Yale_Onc.py)
2. Extract `sas_images` (list of 2D slices with shape (C, H, W))
3. Select middle 3 slices OR all slices (configurable)
4. Stack slices → (C, D, H, W)
5. Use only first image channel (PSIR or first available)
6. Resize to (1, 80, 80, 80)
7. Apply transforms (normalization, augmentation)
8. Load age/ICI via PatientDataExtractor
9. Return (image, label, attributes)

### 4. Training Pipeline (studies/SD_attrivae_age_ici/main.py)

```python
def main():
    # 1. Load organized data
    organized_data = np.load(DATA_FILE, allow_pickle=True).tolist()

    # 2. Load patient metadata extractor
    patient_extractor = PatientDataExtractor(
        excel_file=SPREADSHEET_PATH,
        columns_config={'age': {...}, 'ici': {...}},
        key_column='Anonymized Code'
    )

    # 3. Filter patients with missing age/ICI
    valid_data = [p for p in organized_data if has_valid_attributes(p)]

    # 4. Train/val split (stratified by CHIP_label)
    train_data, val_data = stratified_split(valid_data, val_frac=0.2)

    # 5. Compute normalization stats from training data
    age_stats = compute_age_stats(train_data, patient_extractor)

    # 6. Create datasets and dataloaders
    train_ds = AttriVAEDataset(train_data, patient_extractor, train_transforms,
                                age_mean=age_stats['mean'], age_std=age_stats['std'])
    val_ds = AttriVAEDataset(val_data, patient_extractor, val_transforms,
                              age_mean=age_stats['mean'], age_std=age_stats['std'])

    # 7. Initialize model
    model = ConvVAE(image_channels=1, h_dim=96, latent_size=64, ...)

    # 8. Training loop (adapted from training_main.py)
    for epoch in range(epochs):
        train_loss = train(epoch, model, train_loader, optimizer,
                          use_AR_LOSS=True, num_attributes=2)
        val_loss, acc, auc = test(epoch, model, val_loader)
        # Save checkpoints...
```

### 5. Loss Function Modification (model/loss_functions.py)

Current `reg_loss` iterates over all attribute dimensions and applies regularization to corresponding latent dimensions:
```python
for dim in range(radiomics_.shape[1]):  # radiomics_ has shape (batch, num_attrs)
    x = latent_code[:, dim]  # latent dim
    radiomics_dim = radiomics_[:, dim]  # attribute dim
    AR_loss += reg_loss_sign(x, radiomics_dim, factor=factor)
```

For 2 attributes (age, ICI):
- `attributes[:, 0]` = normalized age → constrain `z[:, 0]`
- `attributes[:, 1]` = ICI (0/1) → constrain `z[:, 1]`

**No modification needed** - the existing `reg_loss` function already handles arbitrary number of attributes.

### 6. Configuration File (studies/SD_attrivae_age_ici/configs/train_attrivae.yaml)

```yaml
# Data
data:
  organized_data_file: "/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2/data/2025-06-01-onc-cohort-144-with-serial-scans-and-103-LGE-masks.npy"
  patient_metadata:
    spreadsheet: "/gpfs/gibbs/project/kwan/jx332/code/2025-09-LGE-CHIP-Classification-V2/link_project_data/CHIP ICI MI CM outcomes - Updated 2025-10-09.xlsx"
    key_column: "Anonymized Code"
    columns:
      age:
        source_column: "age"
        type: int
      ici:
        source_column: "ICI Y (1) or N (0) trial (2)"
        type: category
        mapping: {"0": 0, "1": 1, "2": 1}
  view: "sas"  # Use short-axis only
  slice_selection: "middle_3"  # Use middle 3 slices
  target_size: [80, 80, 80]
  val_fraction: 0.2
  seed: 42

# Model
model:
  image_channels: 1
  h_dim: 96
  latent_size: 64
  n_filters_enc: [8, 16, 32, 64, 2]
  n_filters_dec: [64, 32, 16, 8, 4, 2]

# Training
training:
  batch_size: 8  # Smaller than original 16 due to limited data
  epochs: 1000
  learning_rate: 0.0001

# Loss weights
loss:
  recon_param: 1.0
  beta: 2.0      # KL weight
  alpha: 1.0     # Classification (MLP) weight
  gamma: 10.0    # Attribute regularization weight
  factor: 100.0  # tanh scaling in AR loss

# Flags
use_AR_LOSS: true
is_L1: false

# Output
output_dir: "outputs/attrivae_age_ici"
```

## Implementation Steps

### Phase 1: Data Exploration & Visualization (FIRST)

Before implementing the training pipeline, verify the data is correctly organized.

#### Step 1.1: Create Study Directory
- Create `studies/SD_attrivae_age_ici/` with subdirectories

#### Step 1.2: Create Data Exploration Script
Create `studies/SD_attrivae_age_ici/explore_data.py` to:
1. Load the organized NPY file
2. Check data structure and available fields
3. Verify short-axis (sas) images exist and have expected format
4. Verify CHIP labels are present
5. Load age/ICI from spreadsheet via PatientDataExtractor
6. Report statistics: number of patients, age distribution, ICI distribution, CHIP label distribution
7. Visualize sample images (middle 3 slices from a few patients)

#### Step 1.3: Create Visualization Notebook (optional)
Create `studies/SD_attrivae_age_ici/notebooks/01_data_exploration.ipynb` for interactive exploration

#### Step 1.4: Verify Data Quality
- Check for missing data (patients without sas images, without age/ICI)
- Verify image dimensions and value ranges
- Confirm middle 3 slices selection works correctly

### Phase 2: Dataset Implementation

#### Step 2.1: Implement Dataset Class
- Create `dataset.py` with `AttriVAEDataset`
- Handle slice selection (middle 3)
- Implement 3D volume creation from 2D slices
- Integrate PatientDataExtractor for age/ICI

#### Step 2.2: Test Dataset
- Verify dataset returns correct shapes
- Verify attributes are normalized correctly

### Phase 3: Training Pipeline

#### Step 3.1: Create Main Training Script
- Adapt from existing `training_main.py`
- Add config loading
- Implement train/val split
- Add normalization stats computation

#### Step 3.2: Create Config File
- YAML configuration with all parameters
- Paths to data files

#### Step 3.3: Run Training
- Start with small number of epochs to verify everything works
- Monitor losses (reconstruction, KL, classification, attribute regularization)
- Save checkpoints

## Data Flow Diagram

```
Organized NPY data                PatientDataExtractor
       ↓                                ↓
[sas_images: list of (C,H,W)]    [age, ici from Excel]
       ↓                                ↓
   Select slices                   Normalize age
       ↓                                ↓
Stack → (C, D, H, W)             attributes = [age_norm, ici]
       ↓                                ↓
Resize → (1, 80, 80, 80)                ↓
       ↓                                ↓
   Transforms                           ↓
       ↓                                ↓
       └──────────┬─────────────────────┘
                  ↓
        AttriVAEDataset.__getitem__()
                  ↓
        (image, chip_label, attributes)
                  ↓
               ConvVAE
                  ↓
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
 Decoder      Latent z      MLP Classifier
    ↓             ↓             ↓
Recon Loss   AR Loss      Classification Loss
    ↓        (age→z[0],       ↓
    ↓        ici→z[1])        ↓
    └─────────────┴─────────────┘
                  ↓
            Total Loss
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `studies/SD_attrivae_age_ici/explore_data.py` | Create | **Phase 1**: Data exploration & visualization |
| `studies/SD_attrivae_age_ici/dataset.py` | Create | Phase 2: Custom dataset class |
| `studies/SD_attrivae_age_ici/main.py` | Create | Phase 3: Training entry point |
| `studies/SD_attrivae_age_ici/configs/train_attrivae.yaml` | Create | Phase 3: Configuration |
| `model/model.py` | No change | Use existing ConvVAE |
| `model/loss_functions.py` | No change | Existing reg_loss works for 2 attributes |

## Notes

- The organized data file already contains `CHIP_label` for each patient
- PatientDataExtractor is reused from the existing study
- Age is normalized (z-score), ICI remains binary
- First 2 latent dimensions will be interpretable (age-correlated, ICI-correlated)
- Remaining 62 latent dimensions are unconstrained
