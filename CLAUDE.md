# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Attri-VAE is an attribute-based, disentangled variational autoencoder for interpretable medical image analysis. It learns disentangled representations of 3D medical images (cardiac MRI) where specific latent dimensions correspond to interpretable clinical attributes. The model enables:
- Classification of pathological vs healthy cardiac images
- Latent space manipulation for image generation
- Attribute-wise attention map generation using gradient-weighted activation mapping

Paper: http://arxiv.org/abs/2203.10417

## Running the Code
- note: conda environment LGE should be used for running python scripts.

### Training
```bash
python main.py
```
Training is configured via hyperparameters at the top of `main.py` and `training_main.py`. Key parameters:
- `batch_size`, `epochs`, `learning_rate` - standard training params
- `latent_size` - VAE latent dimension (default: 64)
- `beta`, `alpha`, `gamma`, `factor` - loss weighting coefficients
- `is_L1`, `use_AR_LOSS` - enable L1 regularization and attribute regularization loss

### Testing on ACDC Dataset
```bash
python testing_acdc.py
```

## Architecture

### Core Components

**Model** (`model/model.py`): `ConvVAE` - 3D convolutional VAE with:
- 5-layer 3D conv encoder producing mu/logvar for latent distribution
- 3-layer MLP classifier branching from latent space (binary classification)
- Transposed conv decoder reconstructing 80x80x80 volumes
- Reparameterization trick via both manual implementation and torch.distributions

**Loss Functions** (`model/loss_functions.py`):
- `reconstruction_loss` - MSE (gaussian) or BCE (bernoulli)
- `KL_loss` - KL divergence between posterior and prior
- `mlp_loss_function` - BCE for classification
- `reg_loss` / `reg_loss_sign` - Attribute regularization loss that constrains latent dimensions to correlate with clinical attributes

**Attribute-wise Attention** (`attribute_wise_attention.py`): GradCAM-style attention maps showing which image regions correspond to specific latent dimensions/attributes

### Data Pipeline

**Data loading** (`data_features/`):
1. `data_load_feature_extract.py` - Load EMIDEC cardiac dataset, extract radiomics features
2. `tra_val_split_oversample.py` - Train/val split with optional oversampling
3. `feature_preprocessing.py` - Feature scaling/normalization
4. `feature_selector_analysis.py` - Select attributes for regularization
5. `dataset_dataloader.py` - PyTorch Dataset/DataLoader using MONAI transforms

**Dataset format**: NIfTI (.nii.gz) 3D cardiac MRI volumes. Uses EMIDEC dataset with healthy (N) and myocardial infarction (P) cohorts.

### Key Utilities

- `utils/utils_evaluation.py` - Latent space visualization, interpolation, GradCAM display
- `utils/utils_interpretable_features.py` - Compute myocardial mass/thickness
- `utils/utils_data_process.py` - Clinical info loading, radiomics extraction

## Dependencies

Core: PyTorch, MONAI, nibabel, pyradiomics, scikit-learn, seaborn

## Important Implementation Details

- Input images resized to 80x80x80 with MONAI transforms including contrast adjustment (gamma=5.0) and intensity scaling to [0,1]
- Attribute regularization loss uses pairwise comparison of latent values vs attribute values within batch
- Three model checkpoints saved: best loss, best accuracy, best AUC
- Target layer for attention maps: `conv5_enc` (encoder's last conv layer)
