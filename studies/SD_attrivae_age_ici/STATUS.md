# AttriVAE Age/ICI Study - Current Status

**Last Updated**: 2025-12-28

## Status: TRAINING + VISUALIZATION READY

All components have been implemented and tested. The training pipeline and latent space visualization tools are working.

## Completed Tasks

1. **Study directory structure** - Created at `/gpfs/gibbs/project/kwan/jx332/code/2025-12-Attri-VAE/studies/SD_attrivae_age_ici/`

2. **Implementation plan** - Saved to `PLAN.md`

3. **Data exploration** - `explore_data.py` verified:
   - **115 valid patients** (79.9% of 144 total)
   - SAS images in `sas_segmented_PSIR_images` key
   - 5 slices per patient (median), range 4-7
   - CHIP labels: 61% negative / 39% positive
   - Age: mean 63.7 ± 14.0
   - ICI: 47% no / 53% yes

4. **Dataset class** - `dataset.py`
   - Loads middle 3 slices from short-axis PSIR images
   - Resizes to 80×80×80 volume
   - Normalizes age (z-score), keeps ICI binary
   - Returns (image, label, attributes, subject_id)

5. **Training script** - `main.py`
   - Complete training loop with validation
   - Saves best models (loss, accuracy, AUC)
   - Tested with 2 epochs - working!

6. **Configuration** - `configs/train_attrivae.yaml`

7. **Latent Traversal Visualization** - `visualize_latent_traversal.py`
   - Age traversal: Vary latent dim 0 to see age-related changes
   - ICI traversal: Vary latent dim 1 to see ICI-related changes
   - 2D grid: Age × ICI combined manipulation
   - Latent distribution plots with disentanglement correlation
   - Reconstruction comparison (original vs. decoded)

## Bug Fixes Applied

1. **model/model.py**: Fixed `IMG` → `image_channels` on line 103
2. **main.py**: Added `safe_mean_accuracy` to handle single-class batches

## How to Run Training

```bash
# Full training (500 epochs)
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate LGE
cd /gpfs/gibbs/project/kwan/jx332/code/2025-12-Attri-VAE
python studies/SD_attrivae_age_ici/main.py --config studies/SD_attrivae_age_ici/configs/train_attrivae.yaml

# Quick test (2 epochs)
python studies/SD_attrivae_age_ici/main.py --epochs 2 --config studies/SD_attrivae_age_ici/configs/train_attrivae.yaml

# Custom settings
python studies/SD_attrivae_age_ici/main.py --epochs 100 --batch-size 16 --lr 0.0001
```

## Training Results (2-epoch test)

| Metric | Train | Val |
|--------|-------|-----|
| Loss | 23711 | 12945 |
| Accuracy | 0.71 | 0.75 |
| AUC | 0.57 | 0.68 |
| AR Loss | 17.2 | 12.5 |

## Expected Training Time

- ~90 samples, batch size 8 = ~12 batches per epoch
- ~1-2 minutes per epoch on GPU
- 500 epochs ≈ 8-15 hours

## Output Files

Training outputs saved to `outputs/training/<timestamp>/`:
- `config.yaml` - Configuration used
- `checkpoint.pth` - Latest checkpoint
- `best_loss_model.pth` - Best validation loss
- `best_acc_model.pth` - Best validation accuracy
- `best_auc_model.pth` - Best validation AUC
- `final_model.pth` - Final model
- `history.npy` - Training history

## How to Run Latent Traversal Visualization

After training completes, run the visualization script:

```bash
# Using best AUC model
python studies/SD_attrivae_age_ici/visualize_latent_traversal.py \
    --checkpoint studies/SD_attrivae_age_ici/outputs/training/<timestamp>/best_auc_model.pth \
    --output_dir studies/SD_attrivae_age_ici/outputs/visualizations/<timestamp>

# With custom settings
python studies/SD_attrivae_age_ici/visualize_latent_traversal.py \
    --checkpoint <path_to_model.pth> \
    --n_samples 6 \
    --slice_idx 40
```

### Visualization Outputs

Saved to `outputs/visualizations/<timestamp>/`:
- `reconstruction_comparison.png` - Original vs. reconstructed images
- `latent_distribution.png` - Latent space with disentanglement correlations
- `age_traversal_sample*_<subject_id>.png` - Age manipulation (30→90 years)
- `ici_traversal_sample*_<subject_id>.png` - ICI manipulation (0→1)
- `age_ici_grid_<subject_id>.png` - 2D Age × ICI grid

## Next Steps (Optional)

1. Run full training (500 epochs)
2. Run visualization script on trained model
3. Verify disentanglement via latent distribution correlation plots
4. Generate attention maps for interpretability (use `attribute_wise_attention.py`)
