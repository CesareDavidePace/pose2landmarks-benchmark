#!/usr/bin/env python3
"""
Quick test to verify robustness testing setup works correctly.
Tests with a single model and minimal corruptions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
import numpy as np
from lib.model.pose_baseline import PoseBaselinePL
from lib.data.move4d_data_module import MOVE4DDataModule
from scripts.test_move4d_robustness import (
    CorruptedMOVE4DDataset,
    test_model_with_corruption
)


def main():
    print("=" * 80)
    print("ROBUSTNESS TESTING - DRY RUN")
    print("=" * 80)
    
    # Load checkpoint paths
    ckpt_file = "move4d_checkpoint_paths.json"
    if not os.path.exists(ckpt_file):
        print(f"Error: {ckpt_file} not found. Run extract_checkpoint_paths.py first.")
        sys.exit(1)
    
    with open(ckpt_file, 'r') as f:
        checkpoint_paths = json.load(f)
    
    if len(checkpoint_paths) == 0:
        print("Error: No checkpoints found in JSON file.")
        sys.exit(1)
    
    # Pick first model
    model_name = sorted(checkpoint_paths.keys())[0]
    ckpt_path = checkpoint_paths[model_name]
    
    print(f"\nTesting with model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    
    # Check checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = PoseBaselinePL.load_from_checkpoint(ckpt_path)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Create datamodule
    print("\n2. Creating datamodule...")
    data_root = "data/move4d/MOVE4D"
    split_file = "configs/split/split_lifting_paper_w_val.yaml"
    
    if not os.path.exists(data_root):
        print(f"   ✗ Data root not found: {data_root}")
        sys.exit(1)
    
    if not os.path.exists(split_file):
        print(f"   ✗ Split file not found: {split_file}")
        sys.exit(1)
    
    data_module = MOVE4DDataModule(
        root_dir=data_root,
        batch_size=4,  # Small batch for testing
        num_workers=2,
        n_frames=30,
        split_file=split_file,
        seed=42,
        augmentation_config={"use_augmentation": False},
        anatomical_markers_group="all"
    )
    data_module.setup()
    print(f"   ✓ Test dataset has {len(data_module.test_dataset)} samples")
    
    test_dataset = data_module.test_dataset
    subject_info_dict = test_dataset.subject_info_dict
    
    # Test 1: Baseline (no corruption)
    print("\n3. Testing baseline (no corruption)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    try:
        metrics = test_model_with_corruption(
            model=model,
            test_dataset=test_dataset,
            corruption_type="none",
            corruption_param=0.0,
            batch_size=4,
            num_workers=2,
            device=device,
            subject_info_dict=subject_info_dict,
            seed=42
        )
        print(f"   ✓ Baseline MPJPE: {metrics['mpjpe_mean']:.2f} ± {metrics['mpjpe_std']:.2f} mm")
        print(f"   ✓ Baseline Angle RMSE: {metrics['angle_rmse']:.2f} degrees")
    except Exception as e:
        print(f"   ✗ Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: Gaussian noise
    print("\n4. Testing Gaussian noise (σ=10mm)...")
    sigma_mm = 10.0
    print(f"   σ = {sigma_mm} mm (added in real mm space before normalization)")
    
    try:
        metrics = test_model_with_corruption(
            model=model,
            test_dataset=test_dataset,
            corruption_type="noise",
            corruption_param=sigma_mm,  # Now in mm directly
            batch_size=4,
            num_workers=2,
            device=device,
            subject_info_dict=subject_info_dict,
            seed=42
        )
        print(f"   ✓ Noise MPJPE: {metrics['mpjpe_mean']:.2f} ± {metrics['mpjpe_std']:.2f} mm")
        print(f"   ✓ Noise Angle RMSE: {metrics['angle_rmse']:.2f} degrees")
    except Exception as e:
        print(f"   ✗ Noise test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Dropout
    print("\n5. Testing random dropout (p=0.2)...")
    dropout_rate = 0.2
    
    try:
        metrics = test_model_with_corruption(
            model=model,
            test_dataset=test_dataset,
            corruption_type="dropout",
            corruption_param=dropout_rate,
            batch_size=4,
            num_workers=2,
            device=device,
            subject_info_dict=subject_info_dict,
            seed=42
        )
        print(f"   ✓ Dropout MPJPE: {metrics['mpjpe_mean']:.2f} ± {metrics['mpjpe_std']:.2f} mm")
        print(f"   ✓ Dropout Angle RMSE: {metrics['angle_rmse']:.2f} degrees")
    except Exception as e:
        print(f"   ✗ Dropout test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test corrupted dataset wrapper
    print("\n6. Testing CorruptedMOVE4DDataset wrapper...")
    try:
        corrupted = CorruptedMOVE4DDataset(
            test_dataset,
            corruption_type="noise",
            noise_sigma_mm=5.0,  # Now in mm
            seed=42,
            subject_info_dict=subject_info_dict
        )
        
        # Get a sample
        sample = corrupted[0]
        print(f"   ✓ Sample shape: {sample[0].shape}")
        print(f"   ✓ Wrapper works correctly")
    except Exception as e:
        print(f"   ✗ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYou can now run the full robustness testing:")
    print("  python scripts/test_move4d_robustness.py")
    print()


if __name__ == "__main__":
    main()
