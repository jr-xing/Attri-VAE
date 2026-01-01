# AttriVAE Age/ICI Study - V2 (Two-Stage Training)

This study implements a **two-stage training approach** for AttriVAE:

1. **Stage 1 (Pre-training)**: Train VAE for reconstruction only
2. **Stage 2 (Fine-tuning)**: Add disentanglement losses (MLP + AR)

## Why Two-Stage Training?

The original end-to-end training (v1) failed to learn good reconstructions because:
- Reconstruction loss (~6000) was overwhelmed by KL (~300) and other losses
- The model prioritized latent regularization over image quality

**Solution**: First learn to reconstruct, then add disentanglement.

## Quick Start

```bash
# Activate environment
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate LGE
cd /gpfs/gibbs/project/kwan/jx332/code/2025-12-Attri-VAE

# Stage 1: Pre-train VAE for reconstruction (300 epochs)
python studies/SD_attrivae_age_ici_v2/pretrain_vae.py

# Stage 2: Fine-tune with disentanglement (300 epochs)
python studies/SD_attrivae_age_ici_v2/finetune_attrivae.py \
    --pretrained studies/SD_attrivae_age_ici_v2/outputs/pretrain/<TIMESTAMP>/best_recon_model.pth

# Visualize results
python studies/SD_attrivae_age_ici_v2/visualize.py \
    --checkpoint studies/SD_attrivae_age_ici_v2/outputs/finetune/<TIMESTAMP>/best_auc_model.pth
```

## Directory Structure

```
SD_attrivae_age_ici_v2/
├── pretrain_vae.py       # Stage 1: Pre-train VAE
├── finetune_attrivae.py  # Stage 2: Fine-tune with disentanglement
├── visualize.py          # Visualization script
├── configs/
│   ├── pretrain.yaml     # Stage 1 config
│   └── finetune.yaml     # Stage 2 config
├── outputs/
│   ├── pretrain/         # Stage 1 outputs
│   └── finetune/         # Stage 2 outputs
└── README.md
```

## Stage 1: Pre-training

**Goal**: Learn good image reconstructions.

**Loss**: `recon_loss + beta * kl_loss`
- No MLP classification
- No attribute regularization
- Beta warmup: 0 → 0.01 over 50 epochs

**Output**: `outputs/pretrain/<timestamp>/best_recon_model.pth`

### Key Hyperparameters (pretrain.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 300 | Training epochs |
| learning_rate | 0.001 | Adam LR |
| recon_param | 1.0 | Reconstruction weight (mean reduction) |
| beta | 0.01 | KL weight (with warmup) |
| beta_warmup_epochs | 50 | Warmup period |

## Stage 2: Fine-tuning

**Goal**: Add disentanglement while preserving reconstruction.

**Loss**: `recon_loss + beta * kl_loss + alpha * mlp_loss + gamma * ar_loss`

**Output**: `outputs/finetune/<timestamp>/best_auc_model.pth`

### Key Hyperparameters (finetune.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 300 | Fine-tuning epochs |
| learning_rate | 0.0001 | Lower LR for fine-tuning |
| recon_param | 1.0 | Reconstruction weight |
| beta | 0.01 | KL weight |
| alpha | 1.0 | MLP classification weight |
| gamma | 1.0 | Attribute regularization weight |

## Data

Uses the same dataset as v1:
- **115 valid patients** from oncology cohort
- **Short-axis PSIR images**: Middle 3 slices → 80×80×80 volume
- **Attributes**: Age (normalized), ICI treatment (binary)
- **Classification target**: CHIP status

## Visualization Outputs

After training, the visualization script generates:

| File | Description |
|------|-------------|
| `reconstruction_comparison.png` | Original vs. reconstructed images |
| `latent_distribution.png` | Latent dims vs attributes (correlation) |
| `age_traversal_*.png` | Vary dim 0 (age: 30→90) |
| `ici_traversal_*.png` | Vary dim 1 (ICI: 0→1) |
| `age_ici_grid_*.png` | 2D Age × ICI manipulation |

## Expected Results

### After Stage 1 (Pre-training)
- Reconstruction loss: < 0.01 (MSE with mean reduction)
- Images should look like cardiac MRI

### After Stage 2 (Fine-tuning)
- Reconstruction preserved
- Latent dim 0 correlated with age (high r)
- Latent dim 1 correlated with ICI (high r)
- CHIP classification AUC > 0.6

## Troubleshooting

### Reconstruction still poor after Stage 1
- Increase epochs (try 500)
- Lower beta (try 0.001)
- Increase learning rate (try 0.002)

### Disentanglement not working after Stage 2
- Increase gamma (AR weight)
- Increase epochs
- Check that pre-trained model reconstructs well first

## Related Files

- Dataset: `../SD_attrivae_age_ici/dataset.py`
- Model: `../../model/model.py`
- Loss functions: `../../model/loss_functions.py`
