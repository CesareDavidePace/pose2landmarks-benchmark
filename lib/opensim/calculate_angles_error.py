import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

actions = ["GAIT", "SQUATS", "F-JUMP", "J-JACKS", "T-JUMP", "RUNNING", "JUMP"]

def detect_global_flip(gt, pred):
    """Detect if the entire signal appears to be flipped"""
    corr = np.corrcoef(gt, pred)[0, 1]
    return corr < -0.6  # Threshold can be adjusted

def detect_local_flips(gt, pred, window_size=30):
    """Detect local segments where signals might be flipped relative to each other"""
    n = len(gt)
    flipped_segments = np.zeros(n, dtype=bool)
    
    # Process in windows
    for i in range(0, n, window_size):
        end = min(i + window_size, n)
        seg1 = gt[i:end]
        seg2 = pred[i:end]
        
        # Calculate local correlation
        if np.std(seg1) > 1e-6 and np.std(seg2) > 1e-6:  # Only if there's variation
            corr = np.corrcoef(seg1, seg2)[0, 1]
            if corr < -0.6:  # Threshold can be adjusted
                flipped_segments[i:end] = True
    
    return flipped_segments

def fix_signal(gt, pred, correct_global_flip=True, correct_local_flip=True, correct_offset=True):
    """Apply all corrections to the prediction signal"""
    pred_corrected = pred.copy()
    
    # Track what corrections were applied
    corrections = {
        "global_flipped": False,
        "local_flips": None,
        "offset": 0
    }
    
    # 1. Global flip correction
    if correct_global_flip:
        if detect_global_flip(gt, pred_corrected):
            pred_corrected *= -1
            corrections["global_flipped"] = True
    
    # 2. Local flip correction (only if no global flip was applied)
    if correct_local_flip and not corrections["global_flipped"]:
        local_flips = detect_local_flips(gt, pred_corrected)
        if np.any(local_flips):
            pred_corrected[local_flips] *= -1
            corrections["local_flips"] = local_flips
    
    # 3. Offset correction
    if correct_offset:
        offset = np.mean(gt - pred_corrected)
        pred_corrected += offset
        corrections["offset"] = offset
    
    return pred_corrected, corrections

def compute_errors(gt_df, pred_df, list_angles_error, correct_global_flip=True, 
                  correct_local_flip=True, correct_offset=True, debug=False):
    """
    Compute errors between ground truth and prediction angles with optional corrections.
    
    Args:
        gt_df: DataFrame with ground truth angles
        pred_df: DataFrame with predicted angles
        list_angles_error: List of angle names to compute errors for
        correct_global_flip: Whether to correct global sign flips
        correct_local_flip: Whether to correct local sign flips
        correct_offset: Whether to correct offset
        debug: Whether to print detailed correction information
        
    Returns:
        Dictionary of errors for each angle
    """
    errors = {}
    
    for angle in list_angles_error:
        if angle in gt_df.columns and angle in pred_df.columns:
            gt = gt_df[angle].values
            pred = pred_df[angle].values
            
            # Original RMSE (before any correction)
            original_rmse = np.sqrt(np.mean((gt - pred) ** 2))
            
            # Apply corrections
            pred_corrected, corrections = fix_signal(
                gt, pred, 
                correct_global_flip=correct_global_flip,
                correct_local_flip=correct_local_flip, 
                correct_offset=correct_offset
            )
            
            # Calculate RMSE after corrections
            corrected_rmse = np.sqrt(np.mean((gt - pred_corrected) ** 2))
            
            # Store results
            errors[angle] = {
                "RMSE": corrected_rmse,  # Use corrected RMSE as the primary metric
                "original_RMSE": original_rmse,
                "global_flipped": corrections["global_flipped"],
                "has_local_flips": corrections["local_flips"] is not None and np.any(corrections["local_flips"]),
                "offset": corrections["offset"]
            }
            
            if debug and original_rmse > 15:  # Only report significant improvements
                improvement = ((original_rmse - corrected_rmse) / original_rmse) * 100
                print(f"Angle {angle}:")
                print(f"  - Original RMSE: {original_rmse:.2f}°")
                print(f"  - After corrections: {corrected_rmse:.2f}° (improved by {improvement:.1f}%)")
                print(f"  - Corrections applied: global_flip={corrections['global_flipped']}, "
                      f"local_flips={'Yes' if errors[angle]['has_local_flips'] else 'No'}, "
                      f"offset={corrections['offset']:.2f}°")
        else:
            errors[angle] = {"RMSE": None, "original_RMSE": None, "global_flipped": None, 
                            "has_local_flips": None, "offset": None}

    return errors

def read_mot_file(filepath):
    """Read an OpenSim .mot file and return it as a pandas DataFrame."""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find the start of actual data
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("endheader"):
            start_idx = i + 1
            break

    df = pd.read_csv(filepath, sep="\\s+", skiprows=start_idx)

    return df

def plot_corrections(subject, action, angle, time, gt, pred_original, pred_corrected, 
                     global_flipped, has_local_flips, offset, original_rmse, corrected_rmse):
    """Plot ground truth vs original and corrected predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Original signals
    axes[0].plot(time, gt, label="Ground Truth", linewidth=2)
    axes[0].plot(time, pred_original, label="Original Prediction", linestyle='--')
    axes[0].set_title(f"{subject} - {action} - {angle}: Original (RMSE: {original_rmse:.2f}°)")
    axes[0].set_ylabel("Angle (degrees)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: After all corrections
    flip_type = "Global" if global_flipped else "Local" if has_local_flips else "No"
    axes[1].plot(time, gt, label="Ground Truth", linewidth=2)
    axes[1].plot(time, pred_corrected, label=f"Corrected Prediction", linestyle='--')
    axes[1].set_title(f"After {flip_type} Flip + Offset ({offset:.2f}°) Correction (RMSE: {corrected_rmse:.2f}°)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle (degrees)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join("error_plots", subject, action)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{angle}_correction.png"))
    plt.close()

def run(input_path, output_path, list_angles_error, correct_global_flip=True, 
        correct_local_flip=True, correct_offset=True, plot_high_rmse=False, rmse_threshold=15):
    
    # add iteratively to output_root_path
    iterator = 1
    output_root_path = output_path + "_" + str(iterator)
    while os.path.exists(output_root_path):
        output_root_path = output_path + "_" + str(iterator)
        iterator += 1

    os.makedirs(output_root_path, exist_ok=True)
    
    if plot_high_rmse:
        os.makedirs(os.path.join(output_root_path, "error_plots"), exist_ok=True)

    # Dictionary to store errors
    errors = {}
    
    # List to track high RMSE cases for reporting
    high_rmse_cases = []

    for subject in tqdm(os.listdir(input_path), desc="Processing subjects"):
        if subject == "processing_results.json":
            continue

        subject_path = os.path.join(input_path, subject)

        # Check if the subject folder exists
        if not os.path.exists(subject_path):
            print(f"Subject folder {subject_path} does not exist, skipping.")
            continue

        # Paths for GT and Prediction data
        gt_path = os.path.join(subject_path, "GT", "inverse_kinematics")
        pred_path = os.path.join(subject_path, "PRED", "inverse_kinematics")

        for action in actions:
            file_name = f"{subject}_{action}"
            gt_file_path = os.path.join(gt_path, file_name + "_gt.mot")
            pred_file_path = os.path.join(pred_path, file_name + "_pred.mot")

            # Check if the files exist
            if not os.path.exists(gt_file_path) or not os.path.exists(pred_file_path):
                print(f"Missing files for {action} - {subject}, skipping.")
                continue

            # Read the .mot files
            gt_df = read_mot_file(gt_file_path)
            pred_df = read_mot_file(pred_file_path)

            # Get time vector if needed for plotting
            time = gt_df["time"].values if "time" in gt_df.columns else np.arange(len(gt_df))

            # Drop the 'time' column if it exists
            if "time" in gt_df.columns:
                gt_df = gt_df.drop(columns=["time"])
            if "time" in pred_df.columns:
                pred_df = pred_df.drop(columns=["time"])

            # Compute errors with advanced corrections
            action_errors = compute_errors(
                gt_df, pred_df, list_angles_error,
                correct_global_flip=correct_global_flip,
                correct_local_flip=correct_local_flip,
                correct_offset=correct_offset,
                debug=False
            )

            # Create plots for high RMSE cases if requested
            if plot_high_rmse:
                for angle, metrics in action_errors.items():
                    if metrics["original_RMSE"] is not None and metrics["original_RMSE"] > rmse_threshold:
                        # Track high RMSE cases
                        high_rmse_cases.append({
                            "Subject": subject,
                            "Action": action,
                            "Angle": angle,
                            "Original_RMSE": metrics["original_RMSE"],
                            "Corrected_RMSE": metrics["RMSE"],
                            "Improvement": ((metrics["original_RMSE"] - metrics["RMSE"]) / metrics["original_RMSE"]) * 100
                        })
                        
                        # Plot corrections for this case
                        if "time" in gt_df.columns:
                            gt = gt_df[angle].values
                            pred = pred_df[angle].values
                            pred_corrected, _ = fix_signal(
                                gt, pred, 
                                correct_global_flip=correct_global_flip,
                                correct_local_flip=correct_local_flip, 
                                correct_offset=correct_offset
                            )
                            
                            plot_corrections(
                                subject, action, angle, time, gt, pred, pred_corrected,
                                metrics["global_flipped"], metrics["has_local_flips"], 
                                metrics["offset"], metrics["original_RMSE"], metrics["RMSE"]
                            )

            # Store errors correctly
            if subject not in errors:
                errors[subject] = {}
            if action not in errors[subject]:
                errors[subject][action] = {}

            errors[subject][action] = action_errors

    # Convert errors dictionary to DataFrame
    rows = []
    for subject, actions_data in errors.items():
        for action, angles_data in actions_data.items():
            for angle, metrics in angles_data.items():
                rows.append({
                    "Subject": subject,
                    "Action": action,
                    "Angle": angle,
                    "RMSE": metrics["RMSE"],
                    "Original_RMSE": metrics["original_RMSE"],
                    "Global_Flipped": metrics["global_flipped"],
                    "Has_Local_Flips": metrics["has_local_flips"],
                    "Offset": metrics["offset"]
                })
    
    error_df = pd.DataFrame(rows)
    
    # Save results
    output_file = os.path.join(output_root_path, "angles_error_summary.csv")
    error_df.to_csv(output_file, index=False)
    
    # Save high RMSE cases if any
    if high_rmse_cases:
        high_rmse_df = pd.DataFrame(high_rmse_cases)
        high_rmse_df.to_csv(os.path.join(output_root_path, "high_rmse_cases.csv"), index=False)
        print(f"Found {len(high_rmse_cases)} high RMSE cases (threshold: {rmse_threshold}°)")

    # Calculate metrics by action and angle
    metrics_df = error_df.groupby(["Action", "Angle"]).agg({
        "RMSE": ["mean", "std"],
        "Original_RMSE": ["mean", "std"],
    }).reset_index()



    # Flatten the column hierarchy
    metrics_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col 
        for col in metrics_df.columns.values
    ]
    
    # Calculate improvement percentage
    metrics_df["Improvement_Percent"] = ((metrics_df["Original_RMSE_mean"] - metrics_df["RMSE_mean"]) / 
                                        metrics_df["Original_RMSE_mean"]) * 100
    
    # Save the aggregated metrics
    metrics_file = os.path.join(output_root_path, "angles_error_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Errors saved to {output_file}")
    print(f"Aggregated metrics saved to {metrics_file}")
    
    # Display a summary of the results
    print("\nSummary of results:")
    print(f"Overall mean RMSE before corrections: {error_df['Original_RMSE'].mean():.2f}°")
    print(f"Overall mean RMSE after corrections: {error_df['RMSE'].mean():.2f}°")
    print(f"Overall improvement: {((error_df['Original_RMSE'].mean() - error_df['RMSE'].mean()) / error_df['Original_RMSE'].mean() * 100):.2f}%")
    print(f"Global flips detected: {error_df['Global_Flipped'].sum()}")
    print(f"Local flips detected: {error_df['Has_Local_Flips'].sum()}")
    print(f"Average absolute offset: {error_df['Offset'].abs().mean():.2f}°")


    print("\nDetailed metrics by action and angle:")
    print(metrics_df[["Action", "Angle", "RMSE_mean", "RMSE_std"]])

    # aggregate metrics by action
    action_metrics = error_df.groupby("Action").agg({
        "RMSE": ["mean", "std"],
    }).reset_index()
    print("\nAggregated metrics by action:")
    print(action_metrics)

    # aggregate metrics by suject
    subject_metrics = error_df.groupby("Subject").agg({
        "RMSE": ["mean", "std"],
    }).reset_index()
    print("\nAggregated metrics by subject:")
    print(subject_metrics)

    return output_file, metrics_file

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Calculate errors between ground truth and prediction angles with advanced corrections.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input directory containing GT and Prediction .mot files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the error metrics.")
    parser.add_argument("--disable_global_flip", action="store_true", help="Disable global flip correction.")
    parser.add_argument("--disable_local_flip", action="store_true", help="Disable local flip correction.")
    parser.add_argument("--disable_offset", action="store_true", help="Disable offset correction.")
    parser.add_argument("--plot_high_rmse", action="store_true", help="Generate plots for high RMSE cases.")
    parser.add_argument("--rmse_threshold", type=float, default=15.0, help="RMSE threshold for identifying high error cases.")
    
    args = parser.parse_args()

    list_angles_error = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r", 
                         "hip_flexion_l", "knee_angle_l", "ankle_angle_l"]

    output_file, metrics_file = run(
        args.input_path, 
        args.output_path, 
        list_angles_error,
        correct_global_flip=not args.disable_global_flip,
        correct_local_flip=not args.disable_local_flip,
        correct_offset=not args.disable_offset,
        plot_high_rmse=args.plot_high_rmse,
        rmse_threshold=args.rmse_threshold
    )

    # print parameters used
    print(f"Parameters used:")
    print(f"  - Global flip correction: {'Enabled' if not args.disable_global_flip else 'Disabled'}")
    print(f"  - Local flip correction: {'Enabled' if not args.disable_local_flip else 'Disabled'}")
    print(f"  - Offset correction: {'Enabled' if not args.disable_offset else 'Disabled'}")
    print(f"  - Plot high RMSE cases: {'Enabled' if args.plot_high_rmse else 'Disabled'}")
    print(f"  - RMSE threshold for high RMSE cases: {args.rmse_threshold}°")

    print(f"Errors saved to {output_file}")
    print(f"Aggregated metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()

# Example:
# python calculate_angles_error.py --input_path outputs/opensim/version_001 --output_path outputs/angle_errors --disable_global_flip
