# MOVE4D Robustness Testing

This directory contains scripts for evaluating robustness of trained MOVE4D models under test-time corruptions.

## Scripts

### 1. `extract_checkpoint_paths.py`
Extracts checkpoint paths from training log files (`.out` files).

**Usage:**
```bash
python scripts/extract_checkpoint_paths.py \
    --logs-dir logs/move4d_exps \
    --output move4d_checkpoint_paths.json
```

**Output:**
- `move4d_checkpoint_paths.json`: JSON file mapping model names to checkpoint paths

### 2. `test_move4d_robustness.py`
Main robustness testing script that evaluates all models under:
- **Gaussian noise corruption**: Various noise levels (σ in mm)
- **Random joint dropout**: Various dropout rates with LOCF imputation

**Usage:**

Basic usage (default settings):
```bash
python scripts/test_move4d_robustness.py \
    --ckpt-file move4d_checkpoint_paths.json \
    --data-root data/move4d/MOVE4D \
    --split-file configs/split/split_lifting_paper_w_val.yaml \
    --out-dir results/robustness
```

Advanced usage (custom parameters):
```bash
python scripts/test_move4d_robustness.py \
    --ckpt-file move4d_checkpoint_paths.json \
    --data-root data/move4d/MOVE4D \
    --split-file configs/split/split_lifting_paper_w_val.yaml \
    --sigma-list "0,5,10,20,30,50" \
    --dropout-list "0.1,0.2,0.3,0.4" \
    --n-replicates 10 \
    --batch-size 64 \
    --num-workers 16 \
    --device cuda \
    --out-dir results/robustness \
    --seed 42
```

Test only specific models using pattern matching:
```bash
python scripts/test_move4d_robustness.py \
    --pattern "transformer|lstm|gru" \
    --ckpt-file move4d_checkpoint_paths.json \
    --out-dir results/robustness_rnn
```

**Command-line Arguments:**

Model selection:
- `--ckpt-file`: JSON file with checkpoint paths (default: `move4d_checkpoint_paths.json`)
- `--pattern`: Regex pattern to filter models (e.g., `"transformer|lstm"`)

Data:
- `--data-root`: MOVE4D dataset root (default: `data/move4d/MOVE4D`)
- `--split-file`: Split YAML file (default: `configs/split/split_lifting_paper_w_val.yaml`)

Corruptions:
- `--sigma-list`: Comma-separated noise σ values in mm (default: `"0,5,10,20,30"`)
- `--dropout-list`: Comma-separated dropout rates (default: `"0.1,0.2,0.3"`)
- `-K, --n-replicates`: Monte Carlo replicates for noise (default: 5)

Computation:
- `--batch-size`: Batch size for testing (default: 32)
- `--num-workers`: Dataloader workers (default: 8)
- `--device`: Device to use (default: auto-detect cuda/cpu)
- `--seed`: Random seed (default: 42)

Output:
- `--out-dir`: Output directory (default: `results/robustness`)
- `--no-plots`: Skip generating plots

## Output Files

The script generates the following outputs:

1. **Results CSV**: `results/robustness/move4d_robustness_YYYYMMDD_HHMMSS.csv`
   - Columns: `model`, `corruption_type`, `corruption_param`, `replicate`, `mpjpe_mean`, `mpjpe_std`, `angle_mae`, `angle_rmse`
   
2. **Metadata JSON**: `results/robustness/move4d_robustness_YYYYMMDD_HHMMSS_metadata.json`
   - Contains experiment configuration, git commit, seeds, etc.

3. **Plots** (in `results/robustness/figures/`):
   - `robustness_noise_YYYYMMDD_HHMMSS.png`: MPJPE and angle RMSE vs noise level
   - `robustness_dropout_YYYYMMDD_HHMMSS.png`: MPJPE and angle RMSE vs dropout rate

## Implementation Details

### Noise Corruption
- **Type**: i.i.d. Gaussian noise per coordinate
- **Units**: Specified in mm, converted to normalized units (assuming avg height 1.7m)
- **Application**: Added to input keypoints only; ground truth unchanged
- **Monte Carlo**: Multiple replicates with different random seeds for statistical robustness

### Dropout Corruption
- **Method**: Random per-joint, per-frame dropout
- **Imputation**: LOCF (Last Observation Carried Forward)
  - If joint missing at frame t, use most recent previous value
  - If missing at first frame, keep original value
- **Mask**: Independent dropout decisions per joint and frame

### Metrics
- **MPJPE**: Mean Per-Joint Position Error in mm
  - Computed on all 54 anatomical landmarks
  - Data normalized by subject height, multiplied by 1000 for mm
- **Joint Angles**: ISB-based hip/knee/ankle angles (degrees)
  - MAE: Mean Absolute Error
  - RMSE: Root Mean Square Error

### Data Pipeline
1. Load base MOVE4D test dataset
2. Wrap with `CorruptedMOVE4DDataset` that applies corruptions
3. Use standard PyTorch DataLoader
4. Run model inference in eval mode
5. Compute metrics on predictions vs ground truth

## Example Workflow

```bash
# Step 1: Extract checkpoint paths from log files
python scripts/extract_checkpoint_paths.py

# Step 2: Run robustness tests on all models
python scripts/test_move4d_robustness.py \
    --sigma-list "0,5,10,20,30" \
    --dropout-list "0.1,0.2,0.3" \
    --n-replicates 5 \
    --batch-size 32 \
    --device cuda \
    --out-dir results/robustness

# Step 3: Results are saved automatically
ls results/robustness/
# move4d_robustness_YYYYMMDD_HHMMSS.csv
# move4d_robustness_YYYYMMDD_HHMMSS_metadata.json
# figures/robustness_noise_YYYYMMDD_HHMMSS.png
# figures/robustness_dropout_YYYYMMDD_HHMMSS.png
```

## Notes

- **Reproducibility**: All corruptions use fixed random seeds for reproducibility
- **Normalized units**: Data is height-normalized in preprocessing; noise is converted accordingly
- **Test-time only**: Corruptions applied only at test time; models remain unchanged
- **GPU memory**: Adjust `--batch-size` and `--num-workers` based on available resources
- **Missing models**: If checkpoint not found in logs, model is skipped automatically

---

### 3. `test_move4d_error_propagation.py`
TBME-style error propagation analysis to show that geometric error (MPJPE) is not a sufficient proxy for biomechanical error (angles).

It performs:
- **A) Correlation analysis**: MPJPE (mm) vs angle errors (deg), per sequence and per model.
- **B) Same MPJPE, different angles**: automatically finds a pair of models with similar MPJPE but different angle behavior and plots knee FE trajectories.
- **C) Landmark sensitivity ranking**: correlates per-marker position error with angle error to identify critical landmarks.

**Usage (quick run):**
```bash
python scripts/test_move4d_error_propagation.py \
  --pattern "transformer|lstm|gru" \
  --ckpt-file move4d_checkpoint_paths.json \
  --batch-size 32 \
  --accelerator cuda \
  --devices 0 \
  --tau_mm 1.0 \
  --angle_name knee_flexion \
  --out-dir results/error_prop
```

**Useful options:**
- `--ckpt_dir`: scan all `.ckpt` recursively from a directory (alternative to `--ckpt-file`)
- `--accelerator` and `--devices`: Lightning-like device selection
- `--tau_mm`: tolerance to define “similar MPJPE” model pairs (default `1.0`)
- `--angle_name {knee_flexion,avg_angle}`: angle target for matched-pair selection
- `--do_sensitivity`: enable landmark sensitivity ranking/plot
- `--corruption-type {none,noise,dropout}`: optional stress-test mode
- `--noise-sigma-mm`: used when `--corruption-type noise`
- `--dropout-rate`: used when `--corruption-type dropout`

**Main outputs**:
- `results/error_prop/move4d_errorprop_YYYYMMDD_HHMMSS.csv` (per-window metrics)
- `results/error_prop/summary_YYYYMMDD_HHMMSS.json` (correlations + matched pair info)
- `results/error_prop/marker_errors_YYYYMMDD_HHMMSS.csv`
- `results/error_prop/correlation_summary.csv`
- `results/error_prop/figures/scatter_mpjpe_vs_angle_rmse.png`
- `results/error_prop/figures/same_mpjpe_different_angles_knee_fe.png` (if pair found)
- `results/error_prop/landmark_sensitivity_ranking.csv` (if `--do_sensitivity`)
- `results/error_prop/landmark_sensitivity_top15.png` (if `--do_sensitivity`)

## Troubleshooting

**Issue**: `KeyError` in batch unpacking
- **Solution**: Ensure `collate_fn_corrupted` returns dictionary matching MOVE4D format

**Issue**: Out of memory
- **Solution**: Reduce `--batch-size` or `--num-workers`

**Issue**: Slow angle computation
- **Solution**: Angle errors are optional; main metric is MPJPE. Angle computation can be slow for large batches.

**Issue**: NaN in angle errors
- **Solution**: Some frames may fail angle computation (e.g., degenerate marker configurations). These are skipped.
