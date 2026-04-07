#!/usr/bin/env python3
"""
MOVE4D Error Propagation Analysis (TBME-style)
==============================================

Goal:
1) Show geometric error (MPJPE, mm) does not fully explain biomechanical error (angles, deg)
2) Find "same MPJPE, different angles" model pair
3) Rank landmark sensitivity to angle errors
"""

import os
import sys
import json
import argparse
import warnings
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.model.pose_baseline import PoseBaselinePL
from lib.data.move4d_data_module import MOVE4DDataModule
from lib.utils.metrics import BodyAngleCalculator
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
    Test-time corruption wrapper used for optional noise/dropout analyses.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corruption_type: str = "none",
        noise_sigma_mm: float = 0.0,
        dropout_rate: float = 0.0,
        seed: int = 42,
        subject_info_dict: Optional[Dict] = None,
    ):
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
        kp3d, al, subject, action, fps = self.base_dataset[idx]

        subject_str = subject[0] if isinstance(subject, np.ndarray) else subject
        if self.subject_info_dict and subject_str in self.subject_info_dict:
            height_m = self.subject_info_dict[subject_str]["height_cm"] / 100.0
        else:
            height_m = 1.7

        kp3d_corrupted = self._apply_corruption(kp3d.numpy(), height_m)
        return torch.from_numpy(kp3d_corrupted).float(), al, subject, action, fps

    def _apply_corruption(self, kp3d: np.ndarray, height_m: float) -> np.ndarray:
        if self.corruption_type == "none":
            return kp3d

        kp3d = kp3d.copy()
        if self.corruption_type == "noise":
            return self._add_gaussian_noise(kp3d, height_m)
        if self.corruption_type == "dropout":
            return self._apply_dropout_with_locf(kp3d)
        raise ValueError(f"Unknown corruption type: {self.corruption_type}")

    def _add_gaussian_noise(self, kp3d: np.ndarray, height_m: float) -> np.ndarray:
        kp3d_mm = kp3d * height_m * 1000.0
        noise_mm = self.rng.randn(*kp3d.shape) * self.noise_sigma_mm
        kp3d_noisy_mm = kp3d_mm + noise_mm
        return kp3d_noisy_mm / 1000.0 / height_m

    def _apply_dropout_with_locf(self, kp3d: np.ndarray) -> np.ndarray:
        t_len, n_joints, _ = kp3d.shape
        keep_mask = self.rng.rand(t_len, n_joints) > self.dropout_rate
        kp3d_imputed = kp3d.copy()

        for j in range(n_joints):
            for t in range(t_len):
                if keep_mask[t, j]:
                    continue
                if t == 0:
                    continue
                prev_t = t - 1
                while prev_t >= 0 and not keep_mask[prev_t, j]:
                    prev_t -= 1
                if prev_t >= 0:
                    kp3d_imputed[t, j] = kp3d_imputed[prev_t, j]

        return kp3d_imputed


def collate_fn_corrupted(batch):
    kp3d_list, al_list, subjects, actions, fps_list = zip(*batch)
    return {
        "kp3d": torch.stack(kp3d_list),
        "al": torch.stack(al_list),
        "subject": list(subjects),
        "action": list(actions),
        "fps": list(fps_list),
    }


def parse_device(device: str) -> str:
    if device.isdigit():
        device = f"cuda:{device}"
    elif device == "cuda":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cpu" or device.startswith("cuda:"):
        pass
    else:
        warnings.warn(f"Unknown device format '{device}', using auto")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, using CPU")
            return "cpu"
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        if gpu_id >= torch.cuda.device_count():
            warnings.warn(f"GPU {gpu_id} not available, using cuda:0")
            return "cuda:0"
    return device


def parse_device_from_accelerator(accelerator: str, devices: str, fallback_device: str) -> str:
    """
    Parse PL-like accelerator/devices arguments into a single torch device string.
    """
    if accelerator == "cpu":
        return "cpu"

    dev = str(devices).strip()
    if dev == "":
        return parse_device(fallback_device)

    # Accept formats like "0", "1", "cuda:0", "[0]", "0,1" (pick first deterministically).
    dev = dev.strip("[]")
    if "," in dev:
        dev = dev.split(",")[0].strip()
    return parse_device(dev)


def load_checkpoint_paths(ckpt_file: Optional[str], ckpt_dir: Optional[str], pattern: Optional[str]) -> Dict[str, str]:
    """
    Load checkpoints either from mapping json or by scanning a directory.
    """
    checkpoint_paths: Dict[str, str] = {}

    if ckpt_file and os.path.exists(ckpt_file):
        with open(ckpt_file, "r") as f:
            checkpoint_paths = json.load(f)
    elif ckpt_dir:
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True))
        for p in ckpt_files:
            model_name = Path(p).parent.parent.name if Path(p).parent.parent else Path(p).stem
            if model_name in checkpoint_paths:
                model_name = f"{model_name}__{Path(p).stem}"
            checkpoint_paths[model_name] = p
    else:
        raise FileNotFoundError("Provide --ckpt-file or --ckpt_dir")

    if pattern:
        import re
        rgx = re.compile(pattern)
        checkpoint_paths = {k: v for k, v in checkpoint_paths.items() if rgx.search(k)}

    return checkpoint_paths


def subject_to_str(subject) -> str:
    if isinstance(subject, np.ndarray):
        return str(subject[0]) if subject.size > 0 else "unknown_subject"
    if isinstance(subject, (list, tuple)):
        return str(subject[0]) if len(subject) > 0 else "unknown_subject"
    return str(subject)


def action_to_str(action) -> str:
    if isinstance(action, np.ndarray):
        return str(action[0]) if action.size > 0 else "unknown_action"
    if isinstance(action, (list, tuple)):
        return str(action[0]) if len(action) > 0 else "unknown_action"
    return str(action)


def build_markers(seq_tensor: torch.Tensor, frame_idx: int, marker_names_subset: List[str]) -> Dict[str, np.ndarray]:
    return {name: seq_tensor[frame_idx, i].numpy() for i, name in enumerate(marker_names_subset)}


def compute_angle_metrics_for_sequence(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    marker_names_subset: List[str],
    angle_calc: BodyAngleCalculator,
) -> Dict[str, object]:
    all_abs_errors = []
    knee_fe_pred = []
    knee_fe_gt = []
    knee_fe_abs = []

    for frame_idx in range(pred_seq.shape[0]):
        pred_markers = build_markers(pred_seq, frame_idx, marker_names_subset)
        gt_markers = build_markers(gt_seq, frame_idx, marker_names_subset)

        try:
            angles_pred = angle_calc.compute_angles(pred_markers)
            angles_gt = angle_calc.compute_angles(gt_markers)
        except Exception:
            knee_fe_pred.append(np.nan)
            knee_fe_gt.append(np.nan)
            knee_fe_abs.append(np.nan)
            continue

        frame_errors = []
        for joint_name in ("hip", "knee", "ankle"):
            for angle_name in ("FE", "AB-AD", "ROT"):
                err = np.abs(angles_gt[joint_name][angle_name] - angles_pred[joint_name][angle_name])
                if np.isfinite(err):
                    all_abs_errors.append(float(err))
                    frame_errors.append(float(err))

        knee_gt = angles_gt["knee"]["FE"]
        knee_pred = angles_pred["knee"]["FE"]
        if np.isfinite(knee_gt) and np.isfinite(knee_pred):
            knee_fe_gt.append(float(knee_gt))
            knee_fe_pred.append(float(knee_pred))
            knee_fe_abs.append(float(np.abs(knee_gt - knee_pred)))
        else:
            knee_fe_gt.append(np.nan)
            knee_fe_pred.append(np.nan)
            knee_fe_abs.append(np.nan)

    if len(all_abs_errors) == 0:
        angle_mae = np.nan
        angle_rmse = np.nan
    else:
        e = np.asarray(all_abs_errors, dtype=np.float64)
        angle_mae = float(np.mean(e))
        angle_rmse = float(np.sqrt(np.mean(np.square(e))))

    knee_valid = np.asarray([x for x in knee_fe_abs if np.isfinite(x)], dtype=np.float64)
    knee_fe_rmse = float(np.sqrt(np.mean(np.square(knee_valid)))) if knee_valid.size > 0 else np.nan

    return {
        "angle_mae_deg": angle_mae,
        "angle_rmse_deg": angle_rmse,
        "knee_fe_rmse_deg": knee_fe_rmse,
        "knee_fe_gt_deg": knee_fe_gt,
        "knee_fe_pred_deg": knee_fe_pred,
    }


def evaluate_model_per_sequence(
    model: pl.LightningModule,
    model_name: str,
    test_loader: DataLoader,
    subject_info_dict: Dict,
    device: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Dict[str, List[float]]]]:
    model.eval()
    model.to(device)

    seq_records = []
    marker_records = []
    angle_traj = {}
    angle_calc = BodyAngleCalculator(filter_angles=False)

    warned_shape = False
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Per-seq eval: {model_name}", leave=False):
            kp3d_batch = batch["kp3d"].to(device)
            gt_batch = batch["al"].to(device)
            pred_batch = model(kp3d_batch)

            batch_size = pred_batch.shape[0]
            n_landmarks = pred_batch.shape[2]

            if n_landmarks > len(markers_names):
                if not warned_shape:
                    warnings.warn(
                        f"{model_name}: outputs {n_landmarks} landmarks but only {len(markers_names)} names exist. "
                        "Angles will be NaN."
                    )
                    warned_shape = True
                marker_names_subset = []
                angle_enabled = False
            else:
                marker_names_subset = markers_names[:n_landmarks]
                missing_required = [m for m in REQUIRED_ANGLE_MARKERS if m not in marker_names_subset]
                angle_enabled = len(missing_required) == 0
                if not angle_enabled and not warned_shape:
                    warnings.warn(
                        f"{model_name}: missing required angle markers ({', '.join(missing_required)}). "
                        "Angles will be NaN."
                    )
                    warned_shape = True

            for b in range(batch_size):
                subject = subject_to_str(batch["subject"][b])
                action = action_to_str(batch["action"][b])
                seq_uid = f"{subject}__{action}__clip{sample_idx:05d}"
                trial_id = f"{subject}__{action}"

                height_m = subject_info_dict.get(subject, {}).get("height_cm", 170.0) / 100.0

                pred_seq = pred_batch[b].detach().cpu()
                gt_seq = gt_batch[b].detach().cpu()

                # MPJPE in mm, denormalized with subject-specific height.
                per_marker = torch.sqrt(((pred_seq - gt_seq) ** 2).sum(dim=-1))  # (T, L)
                per_marker_mm = per_marker.numpy() * height_m * 1000.0
                mpjpe_mm = float(np.mean(per_marker_mm))

                # Marker-wise errors (for sensitivity analysis).
                marker_err_mm = np.mean(per_marker_mm, axis=0)  # (L,)
                for i, marker_name in enumerate(marker_names_subset):
                    marker_records.append(
                        {
                            "model": model_name,
                            "seq_uid": seq_uid,
                            "subject": subject,
                            "action": action,
                            "marker_idx": i,
                            "marker_name": marker_name,
                            "marker_error_mm": float(marker_err_mm[i]),
                        }
                    )

                # Angle metrics.
                if angle_enabled:
                    angle_out = compute_angle_metrics_for_sequence(
                        pred_seq=pred_seq,
                        gt_seq=gt_seq,
                        marker_names_subset=marker_names_subset,
                        angle_calc=angle_calc,
                    )
                    angle_mae_deg = angle_out["angle_mae_deg"]
                    angle_rmse_deg = angle_out["angle_rmse_deg"]
                    knee_fe_rmse_deg = angle_out["knee_fe_rmse_deg"]
                    angle_traj[(model_name, seq_uid)] = {
                        "knee_fe_gt_deg": angle_out["knee_fe_gt_deg"],
                        "knee_fe_pred_deg": angle_out["knee_fe_pred_deg"],
                    }
                else:
                    angle_mae_deg = np.nan
                    angle_rmse_deg = np.nan
                    knee_fe_rmse_deg = np.nan

                seq_records.append(
                    {
                        "model": model_name,
                        "seq_uid": seq_uid,
                        "subject": subject,
                        "trial_id": trial_id,
                        "action": action,
                        "motion_category": action,
                        "window_index": int(sample_idx),
                        "mpjpe_mm": mpjpe_mm,
                        "angle_mae_deg": angle_mae_deg,
                        "angle_rmse_deg": angle_rmse_deg,
                        "knee_fe_rmse_deg": knee_fe_rmse_deg,
                        "n_frames": int(pred_seq.shape[0]),
                        "n_landmarks": int(pred_seq.shape[1]),
                    }
                )
                sample_idx += 1

    return pd.DataFrame(seq_records), pd.DataFrame(marker_records), angle_traj


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan
    p_r, p_p = pearsonr(x, y)
    s_r, s_p = spearmanr(x, y)
    return float(p_r), float(p_p), float(s_r), float(s_p)


def run_correlation_analysis(df_seq: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    rows = []
    for model in sorted(df_seq["model"].unique()):
        d = df_seq[df_seq["model"] == model]
        x = d["mpjpe_mm"].to_numpy()
        y_rmse = d["angle_rmse_deg"].to_numpy()
        y_mae = d["angle_mae_deg"].to_numpy()

        p_r_rmse, p_p_rmse, s_r_rmse, s_p_rmse = safe_corr(x, y_rmse)
        p_r_mae, p_p_mae, s_r_mae, s_p_mae = safe_corr(x, y_mae)

        rows.append(
            {
                "model": model,
                "n_sequences": int(len(d)),
                "pearson_r_mpjpe_vs_angle_rmse": p_r_rmse,
                "pearson_p_mpjpe_vs_angle_rmse": p_p_rmse,
                "spearman_rho_mpjpe_vs_angle_rmse": s_r_rmse,
                "spearman_p_mpjpe_vs_angle_rmse": s_p_rmse,
                "pearson_r_mpjpe_vs_angle_mae": p_r_mae,
                "pearson_p_mpjpe_vs_angle_mae": p_p_mae,
                "spearman_rho_mpjpe_vs_angle_mae": s_r_mae,
                "spearman_p_mpjpe_vs_angle_mae": s_p_mae,
            }
        )

    df_corr = pd.DataFrame(rows)
    corr_path = os.path.join(out_dir, "correlation_summary.csv")
    df_corr.to_csv(corr_path, index=False)
    print(f"Saved: {corr_path}")
    return df_corr


def plot_scatter(df_seq: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df_seq.dropna(subset=["mpjpe_mm", "angle_rmse_deg"]).copy()
    if len(d) == 0:
        print("No valid data for scatter plot.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=d, x="mpjpe_mm", y="angle_rmse_deg", hue="model", alpha=0.65, s=35)
    plt.xlabel("MPJPE (mm)")
    plt.ylabel("Angle RMSE (deg)")
    plt.title("Error Propagation: MPJPE vs Angle RMSE")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "scatter_mpjpe_vs_angle_rmse.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved: {out_path}")


def find_same_mpjpe_pair(
    df_seq: pd.DataFrame,
    tol_mm: float = 1.0,
    angle_name: str = "knee_flexion",
) -> Optional[Tuple[str, str, pd.DataFrame]]:
    angle_col = "knee_fe_rmse_deg" if angle_name == "knee_flexion" else "angle_rmse_deg"
    means = (
        df_seq.groupby("model")[["mpjpe_mm", "angle_rmse_deg", "knee_fe_rmse_deg"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    if len(means) < 2:
        return None

    # Prefer knee flexion if available; otherwise fallback to global angle RMSE.
    if angle_col == "knee_fe_rmse_deg" and means["knee_fe_rmse_deg"].notna().sum() < 2:
        angle_col = "angle_rmse_deg"
    means = means.dropna(subset=["mpjpe_mm", angle_col])
    if len(means) < 2:
        return None

    best = None
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            m1 = means.iloc[i]
            m2 = means.iloc[j]
            mpjpe_gap = abs(m1["mpjpe_mm"] - m2["mpjpe_mm"])
            angle_gap = abs(m1[angle_col] - m2[angle_col])

            if mpjpe_gap <= tol_mm:
                score = angle_gap
                if best is None or score > best[0]:
                    best = (score, m1["model"], m2["model"])

    if best is None:
        # fallback: closest MPJPE pair
        best_gap = None
        chosen = None
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                m1 = means.iloc[i]
                m2 = means.iloc[j]
                mpjpe_gap = abs(m1["mpjpe_mm"] - m2["mpjpe_mm"])
                angle_gap = abs(m1[angle_col] - m2[angle_col])
                if best_gap is None or mpjpe_gap < best_gap:
                    best_gap = mpjpe_gap
                    chosen = (angle_gap, m1["model"], m2["model"])
        if chosen is None:
            return None
        best = chosen

    model_a, model_b = best[1], best[2]
    d_a = df_seq[df_seq["model"] == model_a][
        ["seq_uid", "subject", "action", "mpjpe_mm", "angle_rmse_deg", "knee_fe_rmse_deg"]
    ].rename(
        columns={
            "mpjpe_mm": "mpjpe_a",
            "angle_rmse_deg": "angle_a",
            "knee_fe_rmse_deg": "knee_a",
        }
    )
    d_b = df_seq[df_seq["model"] == model_b][
        ["seq_uid", "mpjpe_mm", "angle_rmse_deg", "knee_fe_rmse_deg"]
    ].rename(
        columns={
            "mpjpe_mm": "mpjpe_b",
            "angle_rmse_deg": "angle_b",
            "knee_fe_rmse_deg": "knee_b",
        }
    )

    merged = d_a.merge(d_b, on="seq_uid", how="inner")
    merged["mpjpe_gap"] = np.abs(merged["mpjpe_a"] - merged["mpjpe_b"])
    merged["knee_gap"] = np.abs(merged["knee_a"] - merged["knee_b"])
    merged["angle_gap"] = np.abs(merged["angle_a"] - merged["angle_b"])
    if angle_col == "knee_fe_rmse_deg":
        merged = merged.dropna(subset=["knee_gap", "mpjpe_gap"])
        sort_col = "knee_gap"
    else:
        merged = merged.dropna(subset=["angle_gap", "mpjpe_gap"])
        sort_col = "angle_gap"
    if len(merged) == 0:
        return None

    near = merged[merged["mpjpe_gap"] <= tol_mm].copy()
    if len(near) == 0:
        near = merged.sort_values("mpjpe_gap").head(20).copy()

    chosen_row = near.sort_values(sort_col, ascending=False).iloc[0]
    return model_a, model_b, chosen_row.to_frame().T


def plot_same_mpjpe_case(
    row_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    angle_traj: Dict[Tuple[str, str], Dict[str, List[float]]],
    out_dir: str,
):
    import matplotlib.pyplot as plt

    row = row_df.iloc[0]
    seq_uid = row["seq_uid"]
    traj_a = angle_traj.get((model_a, seq_uid), None)
    traj_b = angle_traj.get((model_b, seq_uid), None)
    if traj_a is None or traj_b is None:
        print("Could not find angle trajectories for selected case.")
        return

    gt = np.asarray(traj_a["knee_fe_gt_deg"], dtype=float)
    pred_a = np.asarray(traj_a["knee_fe_pred_deg"], dtype=float)
    pred_b = np.asarray(traj_b["knee_fe_pred_deg"], dtype=float)

    if gt.size == 0:
        print("Selected case has empty trajectory.")
        return

    t = np.arange(gt.size)
    plt.figure(figsize=(9, 5))
    plt.plot(t, gt, color="black", linewidth=2.0, label="Ground Truth")
    plt.plot(t, pred_a, linewidth=1.8, label=f"{model_a}")
    plt.plot(t, pred_b, linewidth=1.8, label=f"{model_b}")
    plt.xlabel("Frame")
    plt.ylabel("Knee Flexion-Extension (deg)")
    plt.title(
        f"Same MPJPE, Different Angles\n"
        f"{row['subject']} | {row['action']} | ΔMPJPE={row['mpjpe_gap']:.3f} mm"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "same_mpjpe_different_angles_knee_fe.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")

    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "seq_uid": str(row["seq_uid"]),
        "subject": str(row["subject"]),
        "action": str(row["action"]),
        "mpjpe_a_mm": float(row["mpjpe_a"]),
        "mpjpe_b_mm": float(row["mpjpe_b"]),
        "angle_a_deg": float(row["angle_a"]),
        "angle_b_deg": float(row["angle_b"]),
        "knee_a_deg": float(row["knee_a"]),
        "knee_b_deg": float(row["knee_b"]),
        "delta_mpjpe_mm": float(row["mpjpe_gap"]),
        "delta_angle_rmse_deg": float(row["angle_gap"]),
        "delta_knee_fe_rmse_deg": float(row["knee_gap"]),
    }
    summary_path = os.path.join(out_dir, "same_mpjpe_case_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


def run_sensitivity_analysis(
    df_seq: pd.DataFrame,
    df_marker: pd.DataFrame,
    out_dir: str,
):
    merged = df_marker.merge(
        df_seq[["model", "seq_uid", "angle_rmse_deg", "knee_fe_rmse_deg"]],
        on=["model", "seq_uid"],
        how="left",
    )

    rows = []
    for marker_name, d_m in merged.groupby("marker_name"):
        x = d_m["marker_error_mm"].to_numpy()
        y_angle = d_m["angle_rmse_deg"].to_numpy()
        y_knee = d_m["knee_fe_rmse_deg"].to_numpy()

        p_r_a, p_p_a, s_r_a, s_p_a = safe_corr(x, y_angle)
        p_r_k, p_p_k, s_r_k, s_p_k = safe_corr(x, y_knee)

        rows.append(
            {
                "marker_name": marker_name,
                "n": int(len(d_m)),
                "pearson_r_marker_vs_angle_rmse": p_r_a,
                "pearson_p_marker_vs_angle_rmse": p_p_a,
                "spearman_rho_marker_vs_angle_rmse": s_r_a,
                "spearman_p_marker_vs_angle_rmse": s_p_a,
                "pearson_r_marker_vs_knee_fe_rmse": p_r_k,
                "pearson_p_marker_vs_knee_fe_rmse": p_p_k,
                "spearman_rho_marker_vs_knee_fe_rmse": s_r_k,
                "spearman_p_marker_vs_knee_fe_rmse": s_p_k,
            }
        )

    df_sens = pd.DataFrame(rows)
    df_sens["abs_spearman_angle"] = df_sens["spearman_rho_marker_vs_angle_rmse"].abs()
    df_sens["abs_spearman_knee"] = df_sens["spearman_rho_marker_vs_knee_fe_rmse"].abs()
    df_sens = df_sens.sort_values("abs_spearman_angle", ascending=False)
    out_csv = os.path.join(out_dir, "landmark_sensitivity_ranking.csv")
    df_sens.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot top-k markers.
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_k = df_sens.dropna(subset=["spearman_rho_marker_vs_angle_rmse"]).head(15)
    if len(top_k) > 0:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=top_k,
            x="marker_name",
            y="spearman_rho_marker_vs_angle_rmse",
            color="#2a9d8f",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Spearman rho")
        plt.xlabel("Marker")
        plt.title("Top Landmark Sensitivity to Angle RMSE")
        plt.tight_layout()
        out_plot = os.path.join(out_dir, "landmark_sensitivity_top15.png")
        plt.savefig(out_plot, dpi=170)
        plt.close()
        print(f"Saved: {out_plot}")


def main():
    parser = argparse.ArgumentParser(
        description="Error propagation analysis for MOVE4D (MPJPE vs angle errors)"
    )
    parser.add_argument("--ckpt-file", type=str, default="move4d_checkpoint_paths.json")
    parser.add_argument("-ckpt_dir", "--ckpt_dir", type=str, default=None, help="Directory containing .ckpt files")
    parser.add_argument("--pattern", type=str, default=None, help="Regex filter for model names")
    parser.add_argument("--data-root", type=str, default="data/move4d/MOVE4D")
    parser.add_argument(
        "--split-file",
        type=str,
        default="configs/split/split_lifting_paper_w_val.yaml",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accelerator", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", "--out_dir",
        type=str,
        default="results/error_prop",
    )
    parser.add_argument(
        "-tau_mm", "--tau_mm",
        type=float,
        default=1.0,
        help="Tolerance for selecting model pairs with similar MPJPE",
    )
    parser.add_argument(
        "--angle_name",
        type=str,
        default="knee_flexion",
        choices=["knee_flexion", "avg_angle"],
        help="Angle target used for matched-pair selection and case-study ranking",
    )
    parser.add_argument(
        "--corruption-type",
        type=str,
        default="none",
        choices=["none", "noise", "dropout"],
        help="Optional corruption for stress-testing the propagation analysis",
    )
    parser.add_argument("--noise-sigma-mm", type=float, default=0.0)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--do_sensitivity", action="store_true")
    args = parser.parse_args()

    checkpoint_paths = load_checkpoint_paths(args.ckpt_file, args.ckpt_dir, args.pattern)

    if len(checkpoint_paths) == 0:
        raise RuntimeError("No models selected after filtering.")

    device = parse_device_from_accelerator(args.accelerator, args.devices, args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    per_window_csv = os.path.join(args.out_dir, f"move4d_errorprop_{timestamp}.csv")
    summary_json = os.path.join(args.out_dir, f"summary_{timestamp}.json")

    print("=" * 80)
    print("MOVE4D ERROR PROPAGATION ANALYSIS")
    print("=" * 80)
    print(f"Models: {len(checkpoint_paths)}")
    print(f"Device: {device}")
    print(f"Accelerator/devices: {args.accelerator}/{args.devices}")
    print(f"Corruption: {args.corruption_type}")
    if args.corruption_type == "noise":
        print(f"Noise sigma (mm): {args.noise_sigma_mm}")
    if args.corruption_type == "dropout":
        print(f"Dropout rate: {args.dropout_rate}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 80)

    data_module = MOVE4DDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_frames=30,
        split_file=args.split_file,
        seed=args.seed,
        augmentation_config={"use_augmentation": False},
        anatomical_markers_group="all",
    )
    data_module.setup()
    base_test_dataset = data_module.test_dataset
    subject_info_dict = base_test_dataset.subject_info_dict

    if args.corruption_type == "none":
        test_dataset = base_test_dataset
    elif args.corruption_type == "noise":
        test_dataset = CorruptedMOVE4DDataset(
            base_dataset=base_test_dataset,
            corruption_type="noise",
            noise_sigma_mm=args.noise_sigma_mm,
            seed=args.seed,
            subject_info_dict=subject_info_dict,
        )
    else:
        test_dataset = CorruptedMOVE4DDataset(
            base_dataset=base_test_dataset,
            corruption_type="dropout",
            dropout_rate=args.dropout_rate,
            seed=args.seed,
            subject_info_dict=subject_info_dict,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn_corrupted,
    )

    all_seq = []
    all_marker = []
    all_traj = {}

    for i, (model_name, ckpt_path) in enumerate(sorted(checkpoint_paths.items()), start=1):
        print(f"\n[{i}/{len(checkpoint_paths)}] {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        try:
            model = PoseBaselinePL.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(f"Skip {model_name}: failed to load checkpoint ({e})")
            continue

        df_seq_m, df_marker_m, traj_m = evaluate_model_per_sequence(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            subject_info_dict=subject_info_dict,
            device=device,
        )
        all_seq.append(df_seq_m)
        all_marker.append(df_marker_m)
        all_traj.update(traj_m)

    if len(all_seq) == 0:
        raise RuntimeError("No model successfully evaluated.")

    df_seq = pd.concat(all_seq, ignore_index=True)
    df_marker = pd.concat(all_marker, ignore_index=True)

    df_seq.to_csv(per_window_csv, index=False)
    marker_path = os.path.join(args.out_dir, f"marker_errors_{timestamp}.csv")
    df_marker.to_csv(marker_path, index=False)
    print(f"\nSaved: {per_window_csv}")
    print(f"Saved: {marker_path}")

    df_corr = run_correlation_analysis(df_seq, args.out_dir)
    plot_scatter(df_seq, fig_dir)
    if args.do_sensitivity:
        run_sensitivity_analysis(df_seq, df_marker, args.out_dir)

    pair = find_same_mpjpe_pair(df_seq, tol_mm=args.tau_mm, angle_name=args.angle_name)
    selected_pair_summary = None
    if pair is None:
        print("No valid model pair found for 'same MPJPE, different angles'.")
    else:
        model_a, model_b, row_df = pair
        plot_same_mpjpe_case(row_df, model_a, model_b, all_traj, fig_dir)
        row = row_df.iloc[0]
        selected_pair_summary = {
            "model_a": model_a,
            "model_b": model_b,
            "seq_uid": str(row["seq_uid"]),
            "subject": str(row["subject"]),
            "action": str(row["action"]),
            "mpjpe_a_mm": float(row["mpjpe_a"]),
            "mpjpe_b_mm": float(row["mpjpe_b"]),
            "angle_a_deg": float(row["angle_a"]),
            "angle_b_deg": float(row["angle_b"]),
            "knee_a_deg": float(row["knee_a"]),
            "knee_b_deg": float(row["knee_b"]),
            "delta_mpjpe_mm": float(row["mpjpe_gap"]),
            "delta_angle_rmse_deg": float(row["angle_gap"]),
            "delta_knee_fe_rmse_deg": float(row["knee_gap"]),
        }

    summary = {
        "timestamp": timestamp,
        "seed": args.seed,
        "device": device,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "n_models_requested": len(checkpoint_paths),
        "n_models_evaluated": int(df_seq["model"].nunique()),
        "models_evaluated": sorted(df_seq["model"].unique().tolist()),
        "corruption_type": args.corruption_type,
        "noise_sigma_mm": args.noise_sigma_mm,
        "dropout_rate": args.dropout_rate,
        "tau_mm": args.tau_mm,
        "angle_name": args.angle_name,
        "do_sensitivity": bool(args.do_sensitivity),
        "correlation_per_model": df_corr.to_dict(orient="records"),
        "matched_pair": selected_pair_summary,
        "artifacts": {
            "per_window_csv": per_window_csv,
            "marker_errors_csv": marker_path,
            "scatter_plot": os.path.join(fig_dir, "scatter_mpjpe_vs_angle_rmse.png"),
            "matched_pair_plot": os.path.join(fig_dir, "same_mpjpe_different_angles_knee_fe.png"),
            "sensitivity_csv": os.path.join(args.out_dir, "landmark_sensitivity_ranking.csv") if args.do_sensitivity else None,
            "sensitivity_plot": os.path.join(args.out_dir, "landmark_sensitivity_top15.png") if args.do_sensitivity else None,
        },
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_json}")

    print("\nTop correlation rows (MPJPE vs Angle RMSE):")
    print(df_corr[["model", "pearson_r_mpjpe_vs_angle_rmse", "spearman_rho_mpjpe_vs_angle_rmse"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
