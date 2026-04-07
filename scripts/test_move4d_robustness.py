#!/usr/bin/env python3
"""
Robustness Testing for MOVE4D Models
=====================================

Evaluate all trained models on MOVE4D test set under:
1. Gaussian noise corruption (various sigma values)
2. Random joint dropout / occlusion (various dropout rates)

Uses PyTorch Lightning trainer.test() API with test-time-only corruptions.
Computes MPJPE and joint angle errors (ISB-based).

NOTE: A-POSE sequences (calibration poses) are EXCLUDED from metrics as they
have artificially high errors that skew the results.
"""

import os

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.model.pose_baseline import PoseBaselinePL
from lib.data.move4d_data_module import MOVE4DDataModule
from lib.utils.metrics import BodyAngleCalculator, calculate_rmse
from lib.utils.markers_names_move4d import markers_names

REQUIRED_ANGLE_MARKERS = [
    "Lt_ASIS",
    "Rt_ASIS",
    "Lt_PSIS",
    "Rt_PSIS",
    "Rt_Femoral_Lateral_Epicn",
    "Rt_Femoral_Medial_Epicn",
    "Rt_Medial_Malleolus",
    "Rt_Lateral_Malleolus",
    "Rt_Metatarsal_Phal_I",
    "Suprasternale",
    "Substernale",
]


class CorruptedMOVE4DDataset(Dataset):
    """
    Wrapper dataset that applies test-time corruptions to MOVE4D inputs.
    
    Corruptions:
    - Gaussian noise: i.i.d. per coordinate, specified in MM (added before normalization)
    - Random joint dropout: per-frame joint masking with LOCF imputation
    
    IMPORTANT: Noise is added in real 3D mm space BEFORE normalization,
    then data is re-normalized. This ensures biomechanical validity.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        corruption_type: str = "none",
        noise_sigma_mm: float = 0.0,
        dropout_rate: float = 0.0,
        seed: int = 42,
        subject_info_dict: Optional[Dict] = None
    ):
        """
        Args:
            base_dataset: Underlying MOVE4D dataset
            corruption_type: "noise", "dropout", or "none"
            noise_sigma_mm: Std dev of Gaussian noise in MILLIMETERS (applied in real space)
            dropout_rate: Probability of dropping each joint per frame
            seed: Random seed for reproducibility
            subject_info_dict: Subject info for denormalization/renormalization
        """
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.noise_sigma_mm = noise_sigma_mm
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.subject_info_dict = subject_info_dict
        self.rng = np.random.RandomState(seed)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample (already normalized by subject height)
        kp3d, al, subject, action, fps = self.base_dataset[idx]
        
        # Get subject height for denormalization/renormalization
        subject_str = subject[0] if isinstance(subject, np.ndarray) else subject
        if self.subject_info_dict and subject_str in self.subject_info_dict:
            height_m = self.subject_info_dict[subject_str]["height_cm"] / 100.0
        else:
            height_m = 1.7  # Default average height
        
        # Apply corruption to kp3d only (NOT to ground truth landmarks)
        kp3d_corrupted = self._apply_corruption(kp3d.numpy(), height_m)
        
        return torch.from_numpy(kp3d_corrupted).float(), al, subject, action, fps
    
    def _apply_corruption(self, kp3d: np.ndarray, height_m: float) -> np.ndarray:
        """Apply corruption to input keypoints."""
        if self.corruption_type == "none":
            return kp3d
        
        kp3d = kp3d.copy()  # Don't modify original
        
        if self.corruption_type == "noise":
            return self._add_gaussian_noise(kp3d, height_m)
        
        elif self.corruption_type == "dropout":
            return self._apply_dropout_with_locf(kp3d)
        
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")
    
    def _add_gaussian_noise(self, kp3d: np.ndarray, height_m: float) -> np.ndarray:
        """
        Add i.i.d. Gaussian noise to keypoints in mm space.
        
        CRITICAL: For biomechanical validity, noise is added in real 3D mm space
        BEFORE normalization, then data is re-normalized.
        
        Args:
            kp3d: Normalized keypoints (already divided by height_m)
            height_m: Subject height in meters
        
        Returns:
            Corrupted keypoints (re-normalized)
        """
        # Step 1: Denormalize to real mm space
        kp3d_mm = kp3d * height_m * 1000  # Convert to mm
        
        # Step 2: Add noise in mm space
        noise_mm = self.rng.randn(*kp3d.shape) * self.noise_sigma_mm
        kp3d_noisy_mm = kp3d_mm + noise_mm
        
        # Step 3: Re-normalize
        kp3d_noisy_normalized = kp3d_noisy_mm / 1000 / height_m
        
        return kp3d_noisy_normalized
    
    def _apply_dropout_with_locf(self, kp3d: np.ndarray) -> np.ndarray:
        """
        Apply random joint dropout and impute using LOCF.
        
        IMPORTANT: This implements RANDOM per-(joint, frame) dropout.
        This is NOT biomechanically realistic (real occlusions are structured:
        e.g., entire body parts disappear for consecutive frames).
        
        For more realistic testing, consider implementing:
        - Structured occlusions (e.g., "right leg missing for 20 frames")
        - Camera-based occlusions (e.g., "markers behind body")
        - Self-occlusions during specific movements
        
        Args:
            kp3d: Shape (T, J, 3)
        
        Returns:
            Corrupted keypoints with same shape
        """
        T, J, C = kp3d.shape
        
        # Create dropout mask: True = keep, False = drop. Example: p=0.2 means 20% joints dropped on average.
        # NOTE: This is RANDOM i.i.d. dropout per (joint, frame) - not realistic for real occlusions!
        keep_mask = self.rng.rand(T, J) > self.dropout_rate  # (T, J) 
        
        # Apply LOCF imputation per joint
        kp3d_imputed = kp3d.copy()
        for j in range(J):
            for t in range(T):
                if not keep_mask[t, j]:
                    # Find most recent non-missing value
                    if t == 0:
                        # First frame: keep original (or could use per-joint mean)
                        pass
                    else:
                        # LOCF: use previous frame value
                        prev_t = t - 1
                        while prev_t >= 0 and not keep_mask[prev_t, j]:
                            prev_t -= 1
                        if prev_t >= 0:
                            kp3d_imputed[t, j] = kp3d_imputed[prev_t, j]
                        # else: keep original value at t=0
        
        return kp3d_imputed


def collate_fn_corrupted(batch):
    """Custom collate function for corrupted dataset - matches MOVE4DDataModule format."""
    kp3d_list, al_list, subjects, actions, fps_list = zip(*batch)
    kp3d_batch = torch.stack(kp3d_list)
    al_batch = torch.stack(al_list)
    return {
        "kp3d": kp3d_batch,
        "al": al_batch,
        "subject": list(subjects),
        "action": list(actions),
        "fps": list(fps_list)
    }


def compute_angle_errors_batch(
    pred_batch: torch.Tensor,
    gt_batch: torch.Tensor,
    angle_calculator: BodyAngleCalculator,
    marker_names_subset: List[str],
) -> Dict[str, float]:
    """
    Compute joint angle errors for a batch.
    
    Args:
        pred_batch: (B, T, L, 3) predicted landmarks
        gt_batch: (B, T, L, 3) ground truth landmarks
        angle_calculator: BodyAngleCalculator instance
        marker_names_subset: Marker names aligned to landmark dimension in tensors
    
    Returns:
        Dictionary with mean angle errors across batch
    """
    B, T, L, C = pred_batch.shape
    
    all_errors = []
    error_count = 0
    success_count = 0
    max_error_warnings = 3  # Show a few errors to understand the pattern
    
    for b in range(B):
        for t in range(T):
            try:
                # Build marker dict using only available marker names.
                pred_markers = {
                    name: pred_batch[b, t, i].numpy()
                    for i, name in enumerate(marker_names_subset)
                }
                gt_markers = {
                    name: gt_batch[b, t, i].numpy()
                    for i, name in enumerate(marker_names_subset)
                }
                
                # Compute angles
                angles_pred = angle_calculator.compute_angles(pred_markers)
                angles_gt = angle_calculator.compute_angles(gt_markers)
                
                # Compute errors
                error_dict = angle_calculator.calculate_error_dict(angles_gt, angles_pred)
                
                # Flatten errors
                for joint in error_dict:
                    for angle_type in error_dict[joint]:
                        error_value = error_dict[joint][angle_type]
                        if not np.isnan(error_value):
                            all_errors.append(error_value)
                
                success_count += 1
            
            except Exception as e:
                # Skip frames with computation errors, but warn a few times
                error_count += 1
                if error_count <= max_error_warnings:
                    warnings.warn(f"Angle computation failed for batch {b}, frame {t}: {e}")
                continue
    
    if len(all_errors) == 0:
        return {"angle_mae": np.nan, "angle_rmse": np.nan}
    
    all_errors = np.array(all_errors)
    return {
        "angle_mae": np.mean(all_errors),
        "angle_rmse": np.sqrt(np.mean(all_errors ** 2))
    }


def test_model_with_corruption(
    model: pl.LightningModule,
    test_dataset: Dataset,
    corruption_type: str,
    corruption_param: float,
    batch_size: int,
    num_workers: int,
    device: str,
    subject_info_dict: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Test a model with specified corruption.
    
    Args:
        model: Trained PyTorch Lightning model
        test_dataset: Base MOVE4D test dataset
        corruption_type: "noise" or "dropout"
        corruption_param: sigma_mm for noise, or dropout rate
        batch_size: Batch size for testing
        num_workers: Number of dataloader workers
        device: Device to run on
        subject_info_dict: Subject info dict
        seed: Random seed
    
    Returns:
        Dictionary of metrics
    """
    # Create corrupted dataset
    if corruption_type == "noise":
        corrupted_dataset = CorruptedMOVE4DDataset(
            test_dataset,
            corruption_type="noise",
            noise_sigma_mm=corruption_param,  # Now in mm directly
            seed=seed,
            subject_info_dict=subject_info_dict
        )
    elif corruption_type == "dropout":
        corrupted_dataset = CorruptedMOVE4DDataset(
            test_dataset,
            corruption_type="dropout",
            dropout_rate=corruption_param,
            seed=seed,
            subject_info_dict=subject_info_dict
        )
    else:
        corrupted_dataset = test_dataset
    
    # Create dataloader
    test_loader = DataLoader(
        corrupted_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn_corrupted
    )
    
    # Run inference
    model.eval()
    model.to(device)
    
    all_mpjpe = []
    all_angle_errors = []
    angle_calculator = BodyAngleCalculator(filter_angles=False)
    
    with torch.no_grad():
        angle_warnings_emitted = False
        n_apose_skipped = 0
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {corruption_type}={corruption_param:.3f}", leave=False)):
            kp3d_batch = batch["kp3d"].to(device)
            al_batch = batch["al"].to(device)
            subjects = batch["subject"]
            actions = batch["action"]
            fps_list = batch["fps"]
            
            # Filter out A-POSE sequences (calibration poses with artificially high errors)
            # Convert actions to strings (they might be numpy arrays)
            actions_str = []
            for action in actions:
                if isinstance(action, np.ndarray):
                    action_str = action.item() if action.size == 1 else str(action[0])
                else:
                    action_str = str(action)
                actions_str.append(action_str)
            
            non_apose_mask = [action != "A-POSE" for action in actions_str]
            n_apose_skipped += sum(1 for m in non_apose_mask if not m)
            
            # Skip if entire batch is A-POSE
            if not any(non_apose_mask):
                continue
            
            # Forward pass
            pred_batch = model(kp3d_batch)
            
            # Compute MPJPE (convert back to mm by multiplying with average height)
            # Note: predictions and GT are in normalized units (divided by subject height)
            # For MPJPE, we multiply by 1000 to get mm (assuming unit is meters)
            diff = pred_batch - al_batch
            mpjpe_per_joint = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, T, L)
            mpjpe_batch = mpjpe_per_joint.mean(dim=(1, 2)) * 1000  # (B,) in mm
            
            # Filter MPJPE values for non-A-POSE sequences only
            mpjpe_filtered = [mpjpe for mpjpe, keep in zip(mpjpe_batch.cpu().numpy(), non_apose_mask) if keep]
            all_mpjpe.extend(mpjpe_filtered)
            
            # Compute angle errors (mimic pose_baseline.py logic)
            try:
                n_landmarks = pred_batch.shape[2]
                
                # Only compute angles if model outputs exactly the right number of landmarks
                if n_landmarks != len(markers_names):
                    if not angle_warnings_emitted:
                        warnings.warn(
                            f"Angle computation disabled: model outputs {n_landmarks} landmarks "
                            f"but {len(markers_names)} markers are required for ISB angle computation."
                        )
                        angle_warnings_emitted = True
                else:
                   # We have the right number of landmarks, compute angles
                    # Filter predictions and ground truth for non-A-POSE only
                    pred_filtered = pred_batch.cpu()[non_apose_mask]
                    gt_filtered = al_batch.cpu()[non_apose_mask]
                    
                    if len(pred_filtered) > 0:  # Only compute if we have non-A-POSE samples
                        angle_err = compute_angle_errors_batch(
                            pred_filtered,
                            gt_filtered,
                            angle_calculator,
                            markers_names  # Use full markers_names list
                        )
                        all_angle_errors.append(angle_err)
            except Exception as e:
                if not angle_warnings_emitted:
                    warnings.warn(f"Angle computation failed: {e}")
                    angle_warnings_emitted = True
    
    # Log A-POSE exclusion
    if n_apose_skipped > 0:
        print(f"  → Excluded {n_apose_skipped} A-POSE sequences (calibration poses)")
    
    # Aggregate metrics
    mpjpe_mean = np.mean(all_mpjpe)
    mpjpe_std = np.std(all_mpjpe)
    
    if len(all_angle_errors) > 0:
        angle_mae_list = [e["angle_mae"] for e in all_angle_errors if not np.isnan(e["angle_mae"])]
        angle_rmse_list = [e["angle_rmse"] for e in all_angle_errors if not np.isnan(e["angle_rmse"])]
        angle_mae = np.mean(angle_mae_list) if len(angle_mae_list) > 0 else np.nan
        angle_rmse = np.mean(angle_rmse_list) if len(angle_rmse_list) > 0 else np.nan
    else:
        angle_mae = np.nan
        angle_rmse = np.nan
    
    return {
        "mpjpe_mean": mpjpe_mean,
        "mpjpe_std": mpjpe_std,
        "angle_mae": angle_mae,
        "angle_rmse": angle_rmse
    }


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"


def run_robustness_experiment(
    checkpoint_paths: Dict[str, str],
    data_root: str,
    split_file: str,
    sigma_mm_list: List[float],
    dropout_list: List[float],
    n_replicates: int,
    batch_size: int,
    num_workers: int,
    device: str,
    output_dir: str,
    model_pattern: Optional[str] = None,
    seed: int = 42
):
    """
    Run robustness experiments for all models.
    
    Args:
        checkpoint_paths: Dict mapping model_name -> checkpoint_path
        data_root: Root directory for MOVE4D data
        split_file: Path to split YAML file
        sigma_mm_list: List of noise std devs in mm
        dropout_list: List of dropout rates
        n_replicates: Number of Monte Carlo replicates for noise
        batch_size: Batch size
        num_workers: Number of workers
        device: Device to use
        output_dir: Output directory
        model_pattern: Optional regex pattern to filter models
        seed: Random seed
    """
    import re
    
    # Filter models by pattern if specified
    if model_pattern:
        pattern = re.compile(model_pattern)
        checkpoint_paths = {
            name: path for name, path in checkpoint_paths.items()
            if pattern.search(name)
        }
    
    print(f"Testing {len(checkpoint_paths)} models:")
    for name in sorted(checkpoint_paths.keys()):
        print(f"  - {name}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create datamodule (for getting test dataset)
    data_module = MOVE4DDataModule(
        root_dir=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        n_frames=30,  # Will be overridden by model config
        split_file=split_file,
        seed=seed,
        augmentation_config={"use_augmentation": False},  # No augmentation for test
        anatomical_markers_group="all"
    )
    data_module.setup()
    test_dataset = data_module.test_dataset
    subject_info_dict = test_dataset.subject_info_dict
    
    # Results storage
    results = []
    
    # Test each model
    for model_idx, (model_name, ckpt_path) in enumerate(sorted(checkpoint_paths.items())):
        print(f"\n[{model_idx + 1}/{len(checkpoint_paths)}] Testing model: {model_name}")
        print(f"  Checkpoint: {ckpt_path}")
        
        # Load model
        try:
            model = PoseBaselinePL.load_from_checkpoint(ckpt_path)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            continue
        
        # Test baseline (no corruption)
        print("  Testing baseline (no corruption)...")
        baseline_metrics = test_model_with_corruption(
            model, test_dataset, "none", 0.0,
            batch_size, num_workers, device, subject_info_dict, seed
        )
        results.append({
            "model": model_name,
            "corruption_type": "none",
            "corruption_param": 0.0,
            "replicate": 0,
            **baseline_metrics
        })
        print(f"    MPJPE: {baseline_metrics['mpjpe_mean']:.2f} mm")
        
        # Test with noise
        for sigma_mm in sigma_mm_list:
            if sigma_mm == 0:
                continue  # Already tested baseline
            
            print(f"  Testing Gaussian noise: σ={sigma_mm} mm (added in real mm space before normalization)")
            
            for rep in range(n_replicates):
                rep_seed = seed + rep
                metrics = test_model_with_corruption(
                    model, test_dataset, "noise", sigma_mm,  # Now in mm directly
                    batch_size, num_workers, device, subject_info_dict, rep_seed
                )
                results.append({
                    "model": model_name,
                    "corruption_type": "noise",
                    "corruption_param": sigma_mm,
                    "replicate": rep,
                    **metrics
                })
            
            # Print average across replicates
            rep_mpjpe = [r["mpjpe_mean"] for r in results[-n_replicates:]]
            print(f"    MPJPE (avg over {n_replicates} reps): {np.mean(rep_mpjpe):.2f} ± {np.std(rep_mpjpe):.2f} mm")
        
        # Test with dropout
        for dropout_rate in dropout_list:
            if dropout_rate == 0:
                continue  # Already tested baseline
            
            print(f"  Testing random dropout: p={dropout_rate:.2f}")
            
            metrics = test_model_with_corruption(
                model, test_dataset, "dropout", dropout_rate,
                batch_size, num_workers, device, subject_info_dict, seed
            )
            results.append({
                "model": model_name,
                "corruption_type": "dropout",
                "corruption_param": dropout_rate,
                "replicate": 0,
                **metrics
            })
            print(f"    MPJPE: {metrics['mpjpe_mean']:.2f} mm")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"move4d_robustness_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "seed": seed,
        "n_replicates": n_replicates,
        "sigma_mm_list": sigma_mm_list,
        "dropout_list": dropout_list,
        "data_root": data_root,
        "split_file": split_file,
        "batch_size": batch_size,
        "device": device,
        "n_models": len(checkpoint_paths),
        "models": list(checkpoint_paths.keys())
    }
    metadata_path = os.path.join(output_dir, f"move4d_robustness_{timestamp}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return df, csv_path


def compute_robustness_metrics(df: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Compute robustness metrics: degradation slopes and rankings.
    
    For each model and metric, compute:
    - ΔMetric = Metric(σ_max) - Metric(σ=0)
    - Robustness slope = ΔMetric / σ_max
    
    Lower slope = more robust model
    
    Also computes cross-metric correlation: corr(slope_mpjpe, slope_angle_rmse)
    to assess if landmark error degradation predicts joint angle degradation.
    """
    results = []
    
    # Process noise corruption
    df_noise = df[df["corruption_type"] == "noise"].copy()
    if len(df_noise) > 0:
        # Aggregate over replicates
        df_noise_agg = df_noise.groupby(["model", "corruption_param"]).agg({
            "mpjpe_mean": "mean",
            "angle_mae": "mean",
            "angle_rmse": "mean"
        }).reset_index()
        
        models = sorted(df_noise_agg["model"].unique())
        sigma_values = sorted(df_noise_agg["corruption_param"].unique())
        sigma_max = max([s for s in sigma_values if s > 0])
        
        for model in models:
            df_model = df_noise_agg[df_noise_agg["model"] == model]
            
            # Get baseline (σ=0)
            baseline = df_model[df_model["corruption_param"] == 0.0]
            if len(baseline) == 0:
                continue
            
            baseline_mpjpe = baseline["mpjpe_mean"].values[0]
            baseline_angle_mae = baseline["angle_mae"].values[0]
            baseline_angle_rmse = baseline["angle_rmse"].values[0]
            
            # Get max noise
            max_noise = df_model[df_model["corruption_param"] == sigma_max]
            if len(max_noise) == 0:
                continue
            
            max_mpjpe = max_noise["mpjpe_mean"].values[0]
            max_angle_mae = max_noise["angle_mae"].values[0]
            max_angle_rmse = max_noise["angle_rmse"].values[0]
            
            # Compute degradation and slopes
            delta_mpjpe = max_mpjpe - baseline_mpjpe
            delta_angle_mae = max_angle_mae - baseline_angle_mae
            delta_angle_rmse = max_angle_rmse - baseline_angle_rmse
            
            slope_mpjpe = delta_mpjpe / sigma_max
            slope_angle_mae = delta_angle_mae / sigma_max if not np.isnan(delta_angle_mae) else np.nan
            slope_angle_rmse = delta_angle_rmse / sigma_max if not np.isnan(delta_angle_rmse) else np.nan
            
            results.append({
                "model": model,
                "corruption_type": "noise",
                "baseline_mpjpe": baseline_mpjpe,
                "max_sigma_mm": sigma_max,
                "max_noise_mpjpe": max_mpjpe,
                "delta_mpjpe": delta_mpjpe,
                "slope_mpjpe": slope_mpjpe,
                "baseline_angle_mae": baseline_angle_mae,
                "max_noise_angle_mae": max_angle_mae,
                "delta_angle_mae": delta_angle_mae,
                "slope_angle_mae": slope_angle_mae,
                "baseline_angle_rmse": baseline_angle_rmse,
                "max_noise_angle_rmse": max_angle_rmse,
                "delta_angle_rmse": delta_angle_rmse,
                "slope_angle_rmse": slope_angle_rmse
            })
    
    # Process dropout corruption
    df_dropout = df[df["corruption_type"] == "dropout"].copy()
    if len(df_dropout) > 0:
        models = sorted(df_dropout["model"].unique())
        dropout_values = sorted(df_dropout["corruption_param"].unique())
        dropout_max = max([d for d in dropout_values if d > 0])
        
        for model in models:
            df_model = df_dropout[df_dropout["model"] == model]
            
            # Get baseline from noise results (dropout doesn't have p=0)
            baseline_noise = df[
                ((df["model"] == model) & (df["corruption_type"] == "none")) |
                ((df["model"] == model) & (df["corruption_type"] == "noise") & (df["corruption_param"] == 0.0))
            ]
            if len(baseline_noise) == 0:
                continue
            
            baseline_mpjpe = baseline_noise["mpjpe_mean"].values[0]
            baseline_angle_mae = baseline_noise["angle_mae"].values[0]
            baseline_angle_rmse = baseline_noise["angle_rmse"].values[0]
            
            # Get max dropout
            max_dropout = df_model[df_model["corruption_param"] == dropout_max]
            if len(max_dropout) == 0:
                continue
            
            max_mpjpe = max_dropout["mpjpe_mean"].values[0]
            max_angle_mae = max_dropout["angle_mae"].values[0]
            max_angle_rmse = max_dropout["angle_rmse"].values[0]
            
            # Compute degradation
            delta_mpjpe = max_mpjpe - baseline_mpjpe
            delta_angle_mae = max_angle_mae - baseline_angle_mae
            delta_angle_rmse = max_angle_rmse - baseline_angle_rmse
            
            slope_mpjpe = delta_mpjpe / dropout_max
            slope_angle_mae = delta_angle_mae / dropout_max if not np.isnan(delta_angle_mae) else np.nan
            slope_angle_rmse = delta_angle_rmse / dropout_max if not np.isnan(delta_angle_rmse) else np.nan
            
            results.append({
                "model": model,
                "corruption_type": "dropout",
                "baseline_mpjpe": baseline_mpjpe,
                "max_dropout_rate": dropout_max,
                "max_dropout_mpjpe": max_mpjpe,
                "delta_mpjpe": delta_mpjpe,
                "slope_mpjpe": slope_mpjpe,
                "baseline_angle_mae": baseline_angle_mae,
                "max_dropout_angle_mae": max_angle_mae,
                "delta_angle_mae": delta_angle_mae,
                "slope_angle_mae": slope_angle_mae,
                "baseline_angle_rmse": baseline_angle_rmse,
                "max_dropout_angle_rmse": max_angle_rmse,
                "delta_angle_rmse": delta_angle_rmse,
                "slope_angle_rmse": slope_angle_rmse
            })
    
    # Create dataframe
    df_metrics = pd.DataFrame(results)
    
    # Save to CSV
    metrics_path = os.path.join(output_dir, f"move4d_robustness_metrics_{timestamp}.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"✓ Robustness metrics saved to: {metrics_path}")
    
    # Print rankings
    print("\n" + "="*80)
    print("ROBUSTNESS RANKINGS (Lower slope = More robust)")
    print("="*80)
    
    # Noise robustness
    df_noise_metrics = df_metrics[df_metrics["corruption_type"] == "noise"].copy()
    if len(df_noise_metrics) > 0:
        df_noise_metrics = df_noise_metrics.sort_values("slope_mpjpe")
        print("\n📊 NOISE ROBUSTNESS (MPJPE slope):")
        for idx, row in df_noise_metrics.iterrows():
            print(f"  {idx+1:2d}. {row['model']:25s} | Slope: {row['slope_mpjpe']:6.2f} mm/mm | Δ={row['delta_mpjpe']:6.2f} mm")
        
        # Angle MAE slope
        df_angle = df_noise_metrics.dropna(subset=["slope_angle_mae"]).sort_values("slope_angle_mae")
        if len(df_angle) > 0:
            print("\n📐 NOISE ROBUSTNESS (Angle MAE slope):")
            for idx, row in df_angle.iterrows():
                print(f"  {idx+1:2d}. {row['model']:25s} | Slope: {row['slope_angle_mae']:6.3f} deg/mm | Δ={row['delta_angle_mae']:6.2f} deg")
    
    # Dropout robustness
    df_dropout_metrics = df_metrics[df_metrics["corruption_type"] == "dropout"].copy()
    if len(df_dropout_metrics) > 0:
        df_dropout_metrics = df_dropout_metrics.sort_values("slope_mpjpe")
        print("\n📊 DROPOUT ROBUSTNESS (MPJPE slope):")
        for idx, row in df_dropout_metrics.iterrows():
            print(f"  {idx+1:2d}. {row['model']:25s} | Slope: {row['slope_mpjpe']:7.2f} mm/rate | Δ={row['delta_mpjpe']:6.2f} mm")
    
    print("="*80 + "\n")
    
    # === CROSS-METRIC CORRELATION ANALYSIS ===
    # Does MPJPE slope predict Angle RMSE slope?
    print("\n" + "="*80)
    print("CROSS-METRIC CORRELATION ANALYSIS")
    print("="*80)
    
    df_noise_metrics = df_metrics[df_metrics["corruption_type"] == "noise"].copy()
    if len(df_noise_metrics) > 0:
        # Remove rows with NaN in either slope
        df_valid = df_noise_metrics.dropna(subset=["slope_mpjpe", "slope_angle_rmse"])
        
        if len(df_valid) >= 2:
            from scipy.stats import pearsonr, spearmanr
            
            x = df_valid["slope_mpjpe"].to_numpy()
            y = df_valid["slope_angle_rmse"].to_numpy()
            
            try:
                pearson_r, pearson_p = pearsonr(x, y)
                spearman_r, spearman_p = spearmanr(x, y)
                
                print(f"\n📊 NOISE CORRUPTION: Slope Correlation Analysis")
                print(f"  Question: Does landmark error degradation predict angle error degradation?")
                print(f"  ")
                print(f"  Pearson  r = {pearson_r:.3f}  (p={pearson_p:.4f})")
                print(f"  Spearman ρ = {spearman_r:.3f}  (p={spearman_p:.4f})")
                print(f"  ")
                
                if abs(pearson_r) < 0.5:
                    print(f"  ⚠️  WEAK correlation (|r| < 0.5)")
                    print(f"      → Landmark error degradation is only weakly predictive")
                    print(f"        of joint angle degradation.")
                    print(f"      → This demonstrates that MPJPE alone is insufficient")
                    print(f"        for biomechanical assessment.")
                elif abs(pearson_r) < 0.7:
                    print(f"  ✓  MODERATE correlation (0.5 ≤ |r| < 0.7)")
                    print(f"      → Some relationship between geometric and biomechanical errors.")
                else:
                    print(f"  ✓  STRONG correlation (|r| ≥ 0.7)")
                    print(f"      → Landmark degradation strongly predicts angle degradation.")
                
                # Save correlation to metadata
                corr_summary = {
                    "corruption_type": "noise",
                    "n_models": int(len(df_valid)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "interpretation": "weak" if abs(pearson_r) < 0.5 else ("moderate" if abs(pearson_r) < 0.7 else "strong")
                }
                corr_path = os.path.join(output_dir, f"slope_correlation_{timestamp}.json")
                with open(corr_path, "w") as f:
                    json.dump(corr_summary, f, indent=2)
                print(f"\n  Saved correlation analysis: {corr_path}")
                
            except Exception as e:
                print(f"  ✗ Could not compute correlation: {e}")
        else:
            print(f"  ✗ Not enough valid models (n={len(df_valid)}) for correlation analysis")
    
    # === EFFECT SIZE ANALYSIS ===
    print("\n" + "="*80)
    print("EFFECT SIZE ANALYSIS: Is degradation clinically meaningful?")
    print("="*80)
    
    if len(df_noise_metrics) > 0:
        print(f"\n📐 ANGLE RMSE Degradation (Δ = max_noise - baseline):")
        df_sorted = df_noise_metrics.sort_values("delta_angle_rmse", ascending=False)
        for idx, row in df_sorted.iterrows():
            delta = row["delta_angle_rmse"]
            if np.isnan(delta):
                continue
            
            status = "⚠️  SMALL" if delta < 1.0 else ("✓ MODERATE" if delta < 3.0 else "✓✓ LARGE")
            print(f"  {status:12s} | {row['model']:25s} | Δ = {delta:5.2f}° | σ_max = {row['max_sigma_mm']:.0f} mm")
        
        # Summary statistics
        valid_deltas = df_sorted["delta_angle_rmse"].dropna()
        if len(valid_deltas) > 0:
            print(f"\n  Summary (across {len(valid_deltas)} models):")
            print(f"    Mean Δ angle RMSE: {valid_deltas.mean():.2f}°")
            print(f"    Median:            {valid_deltas.median():.2f}°")
            print(f"    Min:               {valid_deltas.min():.2f}°")
            print(f"    Max:               {valid_deltas.max():.2f}°")
            
            n_small = (valid_deltas < 1.0).sum()
            n_moderate = ((valid_deltas >= 1.0) & (valid_deltas < 3.0)).sum()
            n_large = (valid_deltas >= 3.0).sum()
            
            print(f"\n  Effect size distribution:")
            print(f"    Small (< 1°):     {n_small}/{len(valid_deltas)} models ({100*n_small/len(valid_deltas):.0f}%)")
            print(f"    Moderate (1-3°):  {n_moderate}/{len(valid_deltas)} models ({100*n_moderate/len(valid_deltas):.0f}%)")
            print(f"    Large (≥ 3°):     {n_large}/{len(valid_deltas)} models ({100*n_large/len(valid_deltas):.0f}%)")
            
            if n_small / len(valid_deltas) > 0.5:
                print(f"\n  ⚠️  WARNING: Most models show SMALL degradation (< 1°)")
                print(f"      Reviewers may question clinical relevance.")
                print(f"      Consider:")
                print(f"        - Increasing noise levels (try σ=50-100 mm)")
                print(f"        - Using structured occlusions (not random dropout)")
                print(f"        - Testing on complex movements (JUMP, RUNNING)")
    
    print("="*80 + "\n")
    
    return df_metrics


def plot_robustness_results(df: pd.DataFrame, output_dir: str, timestamp: str):
    """Generate robustness plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Get unique models
    models = sorted(df["model"].unique())
    
    # 1. MPJPE vs Noise
    df_noise = df[df["corruption_type"] == "noise"].copy()
    if len(df_noise) > 0:
        # Aggregate over replicates
        df_noise_agg = df_noise.groupby(["model", "corruption_param"]).agg({
            "mpjpe_mean": ["mean", "std"],
            "angle_rmse": ["mean", "std"]
        }).reset_index()
        df_noise_agg.columns = ["model", "sigma_mm", "mpjpe_mean", "mpjpe_std", "angle_rmse_mean", "angle_rmse_std"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MPJPE vs sigma
        ax = axes[0]
        for model in models:
            df_model = df_noise_agg[df_noise_agg["model"] == model]
            ax.plot(df_model["sigma_mm"], df_model["mpjpe_mean"], marker='o', label=model)
            ax.fill_between(
                df_model["sigma_mm"],
                df_model["mpjpe_mean"] - df_model["mpjpe_std"],
                df_model["mpjpe_mean"] + df_model["mpjpe_std"],
                alpha=0.2
            )
        ax.set_xlabel("Noise σ (mm)")
        ax.set_ylabel("MPJPE (mm)")
        ax.set_title("MPJPE vs Gaussian Noise")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Angle RMSE vs sigma
        ax = axes[1]
        for model in models:
            df_model = df_noise_agg[df_noise_agg["model"] == model]
            # Filter out NaN values
            df_model = df_model.dropna(subset=["angle_rmse_mean"])
            if len(df_model) > 0:
                ax.plot(df_model["sigma_mm"], df_model["angle_rmse_mean"], marker='o', label=model)
                ax.fill_between(
                    df_model["sigma_mm"],
                    df_model["angle_rmse_mean"] - df_model["angle_rmse_std"],
                    df_model["angle_rmse_mean"] + df_model["angle_rmse_std"],
                    alpha=0.2
                )
        ax.set_xlabel("Noise σ (mm)")
        ax.set_ylabel("Angle RMSE (degrees)")
        ax.set_title("Joint Angle RMSE vs Gaussian Noise")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"robustness_noise_{timestamp}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Noise robustness plot saved to: {fig_path}")
        plt.close()
    
    # 2. MPJPE vs Dropout
    df_dropout = df[df["corruption_type"] == "dropout"].copy()
    if len(df_dropout) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MPJPE vs dropout rate
        ax = axes[0]
        for model in models:
            df_model = df_dropout[df_dropout["model"] == model]
            ax.plot(df_model["corruption_param"], df_model["mpjpe_mean"], marker='o', label=model)
        ax.set_xlabel("Dropout Rate")
        ax.set_ylabel("MPJPE (mm)")
        ax.set_title("MPJPE vs Random Joint Dropout")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Angle RMSE vs dropout rate
        ax = axes[1]
        for model in models:
            df_model = df_dropout[df_dropout["model"] == model]
            df_model = df_model.dropna(subset=["angle_rmse"])
            if len(df_model) > 0:
                ax.plot(df_model["corruption_param"], df_model["angle_rmse"], marker='o', label=model)
        ax.set_xlabel("Dropout Rate")
        ax.set_ylabel("Angle RMSE (degrees)")
        ax.set_title("Joint Angle RMSE vs Random Joint Dropout")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"robustness_dropout_{timestamp}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Dropout robustness plot saved to: {fig_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Robustness testing for MOVE4D models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument("--ckpt-file", type=str, default="move4d_checkpoint_paths.json",
                        help="JSON file with checkpoint paths")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Regex pattern to filter models (e.g., 'transformer|lstm')")
    
    # Data
    parser.add_argument("--data-root", type=str, default="data/move4d/MOVE4D",
                        help="MOVE4D dataset root")
    parser.add_argument("--split-file", type=str,
                        default="configs/split/split_lifting_paper_w_val.yaml",
                        help="Split file")
    
    # Corruptions
    parser.add_argument("--sigma-list", type=str, default="5,10,15,20,25,30",
                        help="Comma-separated noise sigmas in mm (try 50,75,100 for larger effects)")
    parser.add_argument("--dropout-list", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated dropout rates (WARNING: uses random per-joint dropout, not structured occlusion)")
    parser.add_argument("-K", "--n-replicates", type=int, default=10,
                        help="Number of Monte Carlo replicates for noise")
    
    # Computation
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (e.g., 'cuda', 'cuda:0', '0', '1', 'cpu')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--out-dir", type=str, default="results/robustness",
                        help="Output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Parse and validate device string
    device = args.device
    if device.isdigit():
        # User passed just a number like "2", convert to "cuda:2"
        device = f"cuda:{device}"
    elif device == "cuda":
        # Use default GPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda:"):
        # Already in correct format
        pass
    elif device == "cpu":
        # CPU is fine
        pass
    else:
        print(f"Warning: Unknown device format '{device}', using 'cuda' if available")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Validate GPU availability
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs), using GPU 0")
                device = "cuda:0"
    
    # Load checkpoint paths
    if not os.path.exists(args.ckpt_file):
        print(f"Error: Checkpoint file not found: {args.ckpt_file}")
        print("Run extract_checkpoint_paths.py first to generate it.")
        sys.exit(1)
    
    with open(args.ckpt_file, 'r') as f:
        checkpoint_paths = json.load(f)
    
    # Parse corruption parameters
    sigma_mm_list = [0.0] + [float(x) for x in args.sigma_list.split(",")]
    dropout_list = [0.0] + [float(x) for x in args.dropout_list.split(",")]
    
    print("=" * 80)
    print("MOVE4D ROBUSTNESS TESTING")
    print("=" * 80)
    print(f"Checkpoint file: {args.ckpt_file}")
    print(f"Data root: {args.data_root}")
    print(f"Split file: {args.split_file}")
    print(f"Noise sigmas (mm): {sigma_mm_list}")
    print(f"Dropout rates: {dropout_list}")
    print(f"Monte Carlo replicates (noise): {args.n_replicates}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 80)
    print()
    
    # Run experiment
    df, csv_path = run_robustness_experiment(
        checkpoint_paths=checkpoint_paths,
        data_root=args.data_root,
        split_file=args.split_file,
        sigma_mm_list=sigma_mm_list,
        dropout_list=dropout_list,
        n_replicates=args.n_replicates,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,  # Use parsed device
        output_dir=args.out_dir,
        model_pattern=args.pattern,
        seed=args.seed
    )
    
    # Compute robustness metrics (slopes, rankings)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_metrics = compute_robustness_metrics(df, args.out_dir, timestamp)
    
    # Generate plots
    if not args.no_plots:
        plot_robustness_results(df, args.out_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
