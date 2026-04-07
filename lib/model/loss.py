import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import numpy as np
import torch.nn.functional as F
from lib.utils.markers_names_move4d import markers_names_lower_limb, markers_names, SEGMENTS
from lib.isb.isb_joint_angle import ISBJointAngleCalculator
import pandas as pd
# Numpy-based errors

# get the indices of the lower limb markers from markers_names
lower_limb_indices = [markers_names.index(name) for name in markers_names_lower_limb]

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def _prepare_mask(mask: torch.Tensor | None, data: torch.Tensor) -> torch.Tensor:
    """Trasforma il mask in shape (B,F) float32 (1=valido)."""
    if mask is None:
        return torch.ones(data.shape[:2], device=data.device, dtype=data.dtype)
    if mask.dim() == 3:  # (B,F,J)
        mask = mask.sum(dim=-1)
    if mask.dim() == 4:  # (B,F,J,1)
        mask = mask.squeeze(-1)
    return mask.float()

def _prepare_joint_weights(joint_weights, num_joints: int, device, dtype) -> torch.Tensor:
    """Returns joint weights as (1,1,J) tensor for broadcasting."""
    if joint_weights is None:
        return torch.ones(1, 1, num_joints, device=device, dtype=dtype)
    if isinstance(joint_weights, torch.Tensor):
        weights = joint_weights.to(device=device, dtype=dtype)
    else:
        weights = torch.tensor(joint_weights, device=device, dtype=dtype)
    return weights.view(1, 1, -1)

# -----------------------------------------------------------------------------
#  Losses
# -----------------------------------------------------------------------------

def loss_bone_length_prior(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor | None = None,
                           segments: List[Tuple[int,int]] = SEGMENTS,
                           to_mm: bool = True) -> torch.Tensor:
    """Bone‑length consistency (∑ L2) in mm.

    Args
    ----
    pred, target : (B,F,J,3)  – coordinate in metri
    mask         : (B,F) / (B,F,J[,_1]) – 0 invalido, 1 valido
    segments     : lista di tuple (idx₁, idx₂)
    to_mm        : converte l'output finale in millimetri
    """
    B, F, J, _ = pred.shape
    m = _prepare_mask(mask, pred)                    # (B,F)

    loss = pred.new_tensor(0.)
    for i,j in segments:
        len_p = torch.norm(pred[:,:,i]-pred[:,:,j], dim=-1)  # (B,F)
        len_t = torch.norm(target[:,:,i]-target[:,:,j], dim=-1)
        diff2 = ((len_p - len_t)**2) * m
        loss += diff2.sum()

    denom = m.sum() * len(segments)
    if denom == 0:
        return pred.new_tensor(0.)
    return (loss / denom) * (1000.0 if to_mm else 1.0)


def loss_segment_length_consistency(pred: torch.Tensor,
                                    mask: torch.Tensor | None = None,
                                    segments: List[Tuple[int,int]] = SEGMENTS,
                                    to_mm: bool = True) -> torch.Tensor:
    """
    Penalizes temporal variation of segment lengths (rigid consistency).

    Args:
        pred: (B,F,J,3) in meters
        mask: (B,F) or (B,F,J[,_1]) valid frames
        segments: list of marker index pairs
        to_mm: scale output to mm
    """
    B, F, J, _ = pred.shape
    if F <= 1:
        return pred.new_tensor(0.0)

    m = _prepare_mask(mask, pred)  # (B,F)
    loss = pred.new_tensor(0.0)

    for i, j in segments:
        seg_len = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)  # (B,F)
        diff = (seg_len[:, 1:] - seg_len[:, :-1]).abs()
        valid = m[:, 1:] * m[:, :-1]
        loss += (diff * valid).sum()

    denom = (m[:, 1:] * m[:, :-1]).sum() * len(segments)
    if denom == 0:
        return pred.new_tensor(0.0)
    return (loss / denom) * (1000.0 if to_mm else 1.0)


def loss_bone_orientation(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor | None = None,
                          segments: List[Tuple[int,int]] = SEGMENTS,
                          to_mm: bool = False) -> torch.Tensor:
    """Orientation loss = 1 − cosθ (media) tra vettori ossei normalizzati.

    Restituisce una quantità *adimensionale* (o mm se to_mm); spesso la si
    pesa con un fattore λ ≈ 2 nella loss totale.
    """
    m = _prepare_mask(mask, pred)  # (B,F)
    loss = pred.new_tensor(0.)

    for i,j in segments:
        v_p = pred[:,:,i] - pred[:,:,j]   # (B,F,3)
        v_t = target[:,:,i] - target[:,:,j]
        v_p = F.normalize(v_p, dim=-1)
        v_t = F.normalize(v_t, dim=-1)
        dot = (v_p * v_t).sum(dim=-1)          # cosθ
        orient_err = (1.0 - dot) * m           # (B,F)
        loss += orient_err.sum()

    denom = m.sum() * len(segments)
    if denom == 0:
        return pred.new_tensor(0.)
    return (loss / denom) * (1000.0 if to_mm else 1.0)

def loss_frequency(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Frequency domain loss using FFT.
    Penalizes differences in the frequency spectrum of the motion to encourage smoothness
    and correct temporal dynamics.
    
    Args:
        pred: (B, F, J, 3)
        target: (B, F, J, 3)
        mask: (B, F) or (B, F, J)
    """
    if mask is not None:
        m = _prepare_mask(mask, pred) # (B, F)
        # Expand mask for broadcasting: (B, F, 1, 1)
        m = m.unsqueeze(-1).unsqueeze(-1)
        pred = pred * m
        target = target * m

    # FFT along time dimension (dim=1)
    # rfft is efficient for real-valued inputs
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    
    # Compute loss in frequency domain
    # We use L1 loss on the complex difference (robust to outliers)
    diff = pred_fft - target_fft
    loss = torch.mean(torch.abs(diff))
    
    return loss

def loss_temporal_consistency(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Temporal consistency loss that penalizes angle velocity and acceleration spikes.
    Uses finite differences to compute 1st and 2nd order derivatives.
    More interpretable and targeted than frequency domain loss.
    
    Args:
        pred: (B, F, J, 3) - predicted joint positions
        target: (B, F, J, 3) - target joint positions  
        mask: (B, F) or (B, F, J) - valid frame mask
    
    Returns:
        Scalar loss combining velocity and acceleration consistency
    """
    B, F, J, C = pred.shape
    
    if F <= 2:
        return pred.new_tensor(0.0)
    
    m = _prepare_mask(mask, pred)  # (B, F)
    
    # Compute velocities (1st order finite difference)
    pred_vel = pred[:, 1:] - pred[:, :-1]  # (B, F-1, J, 3)
    target_vel = target[:, 1:] - target[:, :-1]
    
    # Compute accelerations (2nd order finite difference)
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # (B, F-2, J, 3)
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    
    # Velocity consistency: penalize difference in velocity
    vel_mask = m[:, 1:].unsqueeze(-1).unsqueeze(-1)  # (B, F-1, 1, 1)
    vel_diff = torch.abs(pred_vel - target_vel) * vel_mask
    vel_loss = vel_diff.sum() / (vel_mask.sum() * J * C + 1e-8)
    
    # Acceleration consistency: penalize difference in acceleration (smoothness)
    acc_mask = m[:, 2:].unsqueeze(-1).unsqueeze(-1)  # (B, F-2, 1, 1)
    acc_diff = torch.abs(pred_acc - target_acc) * acc_mask
    acc_loss = acc_diff.sum() / (acc_mask.sum() * J * C + 1e-8)
    
    # Combine with higher weight on acceleration (smoothness more important)
    # Velocity errors are in m/frame, acceleration in m/frame^2
    # Weight acceleration higher to enforce smoothness
    loss = vel_loss + 2.0 * acc_loss
    
    return loss * 1000.0  # Scale to mm for consistency with other losses

def loss_root_mpjpe(pred, target, left_idx=20, right_idx=21, root_target_idx=53, mask=None):
    """
    Compute MPJPE between the estimated root from input (triangulated keypoints)
    and the root marker in the ground-truth anatomical set.

    Args:
        pred (torch.Tensor): Triangulated input keypoints (B, F, J_pred, 3)
        target (torch.Tensor): Ground-truth anatomical markers (B, F, J_target, 3)
        left_idx (int): Index of left hip in input keypoints (e.g., MediaPipe LEFT_HIP)
        right_idx (int): Index of right hip in input keypoints (e.g., MediaPipe RIGHT_HIP)
        root_target_idx (int): Index of root marker in GT (e.g., SACR or PSIS)
        mask (torch.Tensor, optional): Valid frame mask (B, F) or (B, F, 1)

    Returns:
        torch.Tensor: Scalar root MPJPE in millimeters
    """
    # Mid-hip point from input keypoints (approx. root)
    pred_root = (pred[:, :, left_idx, :] + pred[:, :, right_idx, :]) / 2

    # Root joint from anatomical ground-truth
    target_root = target[:, :, root_target_idx, :]

    # L2 distance
    error = torch.norm(pred_root - target_root, dim=-1)  # (B, F)

    # Apply masking if provided
    if mask is not None:
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] > 1:
            # This is a per-joint mask — reduce it to a per-frame mask
            mask = (mask.sum(dim=-1) > 0).float()  # (B, F)
        error = error * mask
        valid_entries = mask.sum()
        return error.sum() / valid_entries * 1000  # meters → mm
    else:
        return error.mean() * 1000  # meters → mm

def masked_mpjpe(pred, target, mask):
    """
    Compute MPJPE while ignoring padded frames.
    Args:
        pred (torch.Tensor): Predicted joint positions (B, F, J_pred, 3)
        target (torch.Tensor): Ground truth joint positions (B, F, J_target, 3)
        mask (torch.Tensor): Valid frames mask (B, F, J, 1) or (B, F)
    Returns:
        torch.Tensor: Mean MPJPE over valid frames.
    """
    
     # Calculate per-joint errors
    errors = torch.norm(pred - target, dim=-1)  # Shape: (B, F, J_target) 
    
    # Ensure mask has the right shape for broadcasting
    if mask.dim() == 2:  # (B, F) -> (B, F, 1)
        mask = mask.unsqueeze(-1)
    
    # Apply mask and calculate masked mean
    errors = errors * mask  # Shape: (B, F, J_target)
    valid_entries = mask.sum()  # Count valid entries
    
    if valid_entries == 0:
        return torch.tensor(0.0, device=pred.device)  # Avoid division by zero  
        
    return errors.sum() / valid_entries * 1000  # 1000 to convert meters to mm



def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))*1000 # 1000 to convert meters to mm
    

def masked_loss_velocity(predicted, target, mask):
    """
    Masked mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    Only computes the loss for joints where the mask is active.
    Args:
        predicted: Tensor of shape (batch_size, seq_len, num_joints, 3)
        target: Tensor of shape (batch_size, seq_len, num_joints, 3)
        mask: Tensor of shape (batch_size, seq_len, num_joints) or (batch_size, seq_len)
    Returns:
        Scalar tensor with the masked mean velocity error
    """
    assert predicted.shape == target.shape
    
    # Handle different mask shapes
    if len(mask.shape) == 2:  # If mask shape is (batch_size, seq_len)
        # Expand mask to match joints dimension
        mask = mask.unsqueeze(-1).expand(-1, -1, predicted.shape[2])
    
    # Now ensure the mask matches the predicted shape except for the last dimension
    assert mask.shape == predicted.shape[:-1], "Mask shape must match predicted shape except for last dimension"
    
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    
    # Calculate velocities
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    
    # For velocity, we need to adjust the mask to match the velocity dimensions
    # We can use the logical AND of consecutive masks to ensure both frames are valid
    velocity_mask = mask[:, 1:] * mask[:, :-1]
    
    # Calculate per-joint errors
    joint_errors = torch.norm(velocity_predicted - velocity_target, dim=-1)
    
    # Apply mask and compute mean
    masked_errors = joint_errors * velocity_mask
    
    # Sum of errors divided by sum of mask (to get mean of only the masked elements)
    # Adding a small epsilon to avoid division by zero
    mask_sum = torch.sum(velocity_mask) + 1e-8
    masked_mean_error = torch.sum(masked_errors) / mask_sum
    
    return masked_mean_error * 1000  # Convert to mm/s

def masked_loss_velocity_weighted(predicted, target, mask, joint_weights=None):
    """
    Weighted masked mean per-joint velocity error.
    joint_weights: list/tensor of shape (J,) with per-joint weights.
    """
    assert predicted.shape == target.shape

    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1).expand(-1, -1, predicted.shape[2])

    assert mask.shape == predicted.shape[:-1], "Mask shape must match predicted shape except for last dimension"

    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)

    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]

    velocity_mask = mask[:, 1:] * mask[:, :-1]
    joint_weights = _prepare_joint_weights(joint_weights, predicted.shape[2], predicted.device, predicted.dtype)

    joint_errors = torch.norm(velocity_predicted - velocity_target, dim=-1)
    weighted_mask = velocity_mask * joint_weights

    mask_sum = torch.sum(weighted_mask) + 1e-8
    weighted_mean_error = torch.sum(joint_errors * weighted_mask) / mask_sum

    return weighted_mean_error * 1000


def masked_loss_acceleration_weighted(predicted, target, mask, joint_weights=None):
    """
    Weighted masked mean per-joint acceleration error using 2nd order differences.
    joint_weights: list/tensor of shape (J,) with per-joint weights.
    """
    assert predicted.shape == target.shape

    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1).expand(-1, -1, predicted.shape[2])

    assert mask.shape == predicted.shape[:-1], "Mask shape must match predicted shape except for last dimension"

    if predicted.shape[1] <= 2:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)

    acc_predicted = predicted[:, 2:] - 2.0 * predicted[:, 1:-1] + predicted[:, :-2]
    acc_target = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]

    acc_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    joint_weights = _prepare_joint_weights(joint_weights, predicted.shape[2], predicted.device, predicted.dtype)

    joint_errors = torch.norm(acc_predicted - acc_target, dim=-1)
    weighted_mask = acc_mask * joint_weights

    mask_sum = torch.sum(weighted_mask) + 1e-8
    weighted_mean_error = torch.sum(joint_errors * weighted_mask) / mask_sum

    return weighted_mean_error * 1000

    
def normalized_masked_loss_velocity(predicted, target, mask, fps=30.0):
    """
    Normalized masked mean per-joint velocity error.
    Similar to masked_loss_velocity but normalizes the velocities by the target velocities
    to make the metric scale-invariant.
    
    Args:
        predicted: Tensor of shape (batch_size, seq_len, num_joints, 3)
        target: Tensor of shape (batch_size, seq_len, num_joints, 3)
        mask: Tensor of shape (batch_size, seq_len, num_joints) or (batch_size, seq_len)
        fps: Float or Tensor of shape (batch_size,) representing frames per second of each sequence
             Default is 30.0 fps if not specified
    
    Returns:
        Scalar tensor with the normalized masked mean velocity error
    """
    assert predicted.shape == target.shape
    
    # Handle different mask shapes
    if len(mask.shape) == 2:  # If mask shape is (batch_size, seq_len)
        # Expand mask to match joints dimension
        mask = mask.unsqueeze(-1).expand(-1, -1, predicted.shape[2])
    
    # Now ensure the mask matches the predicted shape except for the last dimension
    assert mask.shape == predicted.shape[:-1], "Mask shape must match predicted shape except for last dimension"
    
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    
    # Calculate velocities (displacement between consecutive frames)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    
    # For velocity, we need to adjust the mask to match the velocity dimensions
    # We can use the logical AND of consecutive masks to ensure both frames are valid
    velocity_mask = mask[:, 1:] * mask[:, :-1]
    
    # Handle different fps inputs
    if isinstance(fps, (int, float)):
        # If fps is a scalar, use it for all batches
        fps_factor = torch.tensor(fps, device=predicted.device)
    else:
        # If fps is a tensor (batch_size,), expand it properly for broadcasting
        # convert to tensor 
        fps = torch.tensor(fps)
        fps_factor = fps.to(predicted.device)
    
    # Scale velocities by fps to get units in distance/second
    velocity_predicted = velocity_predicted * fps_factor.view(-1, 1, 1, 1)
    velocity_target = velocity_target * fps_factor.view(-1, 1, 1, 1)
    
    # Calculate per-joint velocities magnitudes
    vel_pred_mag = torch.norm(velocity_predicted, dim=-1)
    vel_target_mag = torch.norm(velocity_target, dim=-1)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Calculate the normalized velocity error (difference divided by target magnitude)
    # This makes the metric scale-invariant
    normalized_errors = torch.abs(vel_pred_mag - vel_target_mag) / (vel_target_mag + epsilon)
    
    # Apply mask and compute mean
    masked_errors = normalized_errors * velocity_mask
    
    # Sum of errors divided by sum of mask (to get mean of only the masked elements)
    mask_sum = torch.sum(velocity_mask) + epsilon
    normalized_masked_mean_error = torch.sum(masked_errors) / mask_sum
    
    return normalized_masked_mean_error
    
def _build_move4d_angle_triplets() -> List[Tuple[int, int, int]]:
    name2idx = {name: idx for idx, name in enumerate(markers_names)}
    def idx(name):
        return name2idx.get(name)

    triplets = []
    candidates = [
        # Hip-Knee-Ankle (use trochanterion + femoral epicondyle + malleolus)
        ("Rt_Trochanterion", "Rt_Femoral_Lateral_Epicn", "Rt_Lateral_Malleolus"),
        ("Lt_Trochanterion", "Lt_Femoral_Lateral_Epicn", "Lt_Lateral_Malleolus"),
        # Knee-Ankle-Toe (use metatarsal phal I)
        ("Rt_Femoral_Lateral_Epicn", "Rt_Lateral_Malleolus", "Rt_Metatarsal_Phal_I"),
        ("Lt_Femoral_Lateral_Epicn", "Lt_Lateral_Malleolus", "Lt_Metatarsal_Phal_I"),
        # Knee-Ankle-Calc (alternate distal point)
        ("Rt_Femoral_Lateral_Epicn", "Rt_Lateral_Malleolus", "Rt_Calcaneous_Post"),
        ("Lt_Femoral_Lateral_Epicn", "Lt_Lateral_Malleolus", "Lt_Calcaneous_Post"),
    ]
    for a, b, c in candidates:
        ia, ib, ic = idx(a), idx(b), idx(c)
        if ia is None or ib is None or ic is None:
            continue
        triplets.append((ia, ib, ic))

    return triplets

MOVE4D_FOOT_MARKER_NAMES = [
    "Lt_Calcaneous_Post", "Rt_Calcaneous_Post",
    "Lt_Medial_Malleolus", "Rt_Medial_Malleolus",
    "Lt_Lateral_Malleolus", "Rt_Lateral_Malleolus",
    "Lt_Metatarsal_Phal_I", "Rt_Metatarsal_Phal_I",
    "Lt_Metatarsal_Phal_V", "Rt_Metatarsal_Phal_V",
    "Lt_Digit_II", "Rt_Digit_II",
]
MOVE4D_FOOT_MARKER_IDXS = [markers_names.index(name) for name in MOVE4D_FOOT_MARKER_NAMES if name in markers_names]
    
MOVE4D_ANGLE_TRIPLETS = _build_move4d_angle_triplets()
ANGLE_TRIPLETS = MOVE4D_ANGLE_TRIPLETS

# -----------------------------------------------------------------------------
#  AMASS Dataset Marker Indices (43 markers with arms)
# -----------------------------------------------------------------------------
# AMASS marker order (NO ARMS - 35 markers):
# 0:RASIS, 1:LASIS, 2:RPSIS, 3:LPSIS, 4:RKnee, 5:RMKnee, 6:RAnkle, 7:RMAnkle,
# 8:RToe, 9:R5meta, 10:RCalc, 11:LKnee, 12:LMKnee, 13:LAnkle, 14:LMAnkle,
# 15:LToe, 16:LCalc, 17:L5meta, 18:RShoulder, 19:LShoulder, 20:C7,
# 21:RThigh1, 22:RThigh2, 23:RThigh3, 24:LThigh1, 25:LThigh2, 26:LThigh3,
# 27:RSh1, 28:RSh2, 29:RSh3, 30:LSh1, 31:LSh2, 32:LSh3, 33:RHJC, 34:LHJC

AMASS_NAME2IDX = {
    'RASIS': 0, 'LASIS': 1, 'RPSIS': 2, 'LPSIS': 3,
    'RKnee': 4, 'RMKnee': 5, 'RAnkle': 6, 'RMAnkle': 7, 
    'RToe': 8, 'R5meta': 9, 'RCalc': 10,
    'LKnee': 11, 'LMKnee': 12, 'LAnkle': 13, 'LMAnkle': 14,
    'LToe': 15, 'LCalc': 16, 'L5meta': 17,
    'RShoulder': 18, 'LShoulder': 19, 'C7': 20,
    'RThigh1': 21, 'RThigh2': 22, 'RThigh3': 23,
    'LThigh1': 24, 'LThigh2': 25, 'LThigh3': 26,
    'RSh1': 27, 'RSh2': 28, 'RSh3': 29,
    'LSh1': 30, 'LSh2': 31, 'LSh3': 32,
    'RHJC': 33, 'LHJC': 34
}

AMASS_FOOT_MARKER_NAMES = [
    "RAnkle", "RMAnkle", "RToe", "R5meta", "RCalc",
    "LAnkle", "LMAnkle", "LToe", "L5meta", "LCalc",
]
AMASS_FOOT_MARKER_IDXS = [AMASS_NAME2IDX[name] for name in AMASS_FOOT_MARKER_NAMES if name in AMASS_NAME2IDX]

# AMASS segments for bone length loss
AMASS_SEGMENTS = [
    (AMASS_NAME2IDX['RASIS'], AMASS_NAME2IDX['LASIS']), # Pelvis Width
    (AMASS_NAME2IDX['RHJC'], AMASS_NAME2IDX['RKnee']),  # R Femur
    (AMASS_NAME2IDX['LHJC'], AMASS_NAME2IDX['LKnee']),  # L Femur
    (AMASS_NAME2IDX['RKnee'], AMASS_NAME2IDX['RAnkle']), # R Tibia
    (AMASS_NAME2IDX['LKnee'], AMASS_NAME2IDX['LAnkle']), # L Tibia
    (AMASS_NAME2IDX['RAnkle'], AMASS_NAME2IDX['RToe']),  # R Foot
    (AMASS_NAME2IDX['LAnkle'], AMASS_NAME2IDX['LToe']),  # L Foot
    (AMASS_NAME2IDX['RShoulder'], AMASS_NAME2IDX['LShoulder']), # Shoulder Width
]

# AMASS angle triplets
AMASS_ANGLE_TRIPLETS = [
    # Right knee flex/ext: RHJC – RKnee – RAnkle
    (AMASS_NAME2IDX['RHJC'], AMASS_NAME2IDX['RKnee'], AMASS_NAME2IDX['RAnkle']),
    
    # Left knee flex/ext: LHJC – LKnee – LAnkle
    (AMASS_NAME2IDX['LHJC'], AMASS_NAME2IDX['LKnee'], AMASS_NAME2IDX['LAnkle']),
    
    # Right ankle dorsiflex/plantarflex: RKnee – RAnkle – RToe
    (AMASS_NAME2IDX['RKnee'], AMASS_NAME2IDX['RAnkle'], AMASS_NAME2IDX['RToe']),
    
    # Left ankle dorsiflex/plantarflex: LKnee – LAnkle – LToe
    (AMASS_NAME2IDX['LKnee'], AMASS_NAME2IDX['LAnkle'], AMASS_NAME2IDX['LToe']),
    
    # (optional) use calcaneus instead of toe
    (AMASS_NAME2IDX['RKnee'], AMASS_NAME2IDX['RAnkle'], AMASS_NAME2IDX['RCalc']),
    (AMASS_NAME2IDX['LKnee'], AMASS_NAME2IDX['LAnkle'], AMASS_NAME2IDX['LCalc']),
]

import torch
import math
from typing import List, Tuple

def _joint_angle_proxy(
    coords: torch.Tensor,
    i_prox: int,
    i_joint: int,
    i_dist: int,
    eps: float = 1e-8,
    to_deg: bool = True,
) -> torch.Tensor:
    """
    coords: (B,F,J,3) in metri
    ritorna: (B,F) con angolo in radianti o gradi
    """
    # vettori dei due segmenti
    v1 = coords[..., i_prox, :] - coords[..., i_joint, :]   # (B,F,3)
    v2 = coords[..., i_dist, :] - coords[..., i_joint, :]   # (B,F,3)

    # normalizza
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + eps)

    # coseno angolo
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)      # (B,F)
    theta = torch.acos(cos_theta)                           # radianti

    if to_deg:
        theta = theta * (180.0 / math.pi)
    return theta                                            # (B,F)

def loss_joint_angle_proxy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    triplets: List[Tuple[int, int, int]] = ANGLE_TRIPLETS,
    to_deg: bool = True,
) -> torch.Tensor:
    """
    Loss sugli angoli articolari proxy (MSE degli angoli).

    pred, target : (B,F,J,3) in metri
    mask         : (B,F) o (B,F,J[,_1]) – 0 invalido, 1 valido
    triplets     : lista (proximal, joint, distal)
    """
    assert pred.shape == target.shape, "pred e target devono avere la stessa shape"
    B, F, J, _ = pred.shape

    m = _prepare_mask(mask, pred)  # (B,F), 0/1 come nella tua bone-length

    loss = pred.new_tensor(0.0)
    for (i_prox, i_joint, i_dist) in triplets:
        theta_p = _joint_angle_proxy(pred,   i_prox, i_joint, i_dist, to_deg=to_deg)
        theta_t = _joint_angle_proxy(target, i_prox, i_joint, i_dist, to_deg=to_deg)

        diff2 = ((theta_p - theta_t) ** 2) * m          # (B,F)
        loss += diff2.sum()

    denom = m.sum() * len(triplets)
    if denom == 0:
        return pred.new_tensor(0.0)

    result = loss / denom   # se to_deg=True è in deg^2
    
    # Safety check: return 0 if NaN to prevent gradient explosion
    if torch.isnan(result) or torch.isinf(result):
        return pred.new_tensor(0.0)
    
    return result


# -----------------------------------------------------------------------------
#  ISB Joint Angles
# -----------------------------------------------------------------------------

# Assicurati che ISBJointAngleCalculator sia definito o importato qui
# Se il codice ISB è in isb_joint_angle.py nella stessa directory o in un package:
# from .isb_joint_angle import ISBJointAngleCalculator
# Oppure incolla la definizione della classe ISBJointAngleCalculator direttamente qui
# (come l'hai fornita nell'input)

###############################################################################
# ISBJointAngleCalculator (incollato dall'utente)                             #
###############################################################################
# Helper maths utilities                                                       #
###############################################################################

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _orthonormalise(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _rot_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    R = np.stack([x, y, z], axis=-1)
    return _orthonormalise(R)


###############################################################################
# Euler helpers                                                                #
###############################################################################

def _euler_zyx(R: np.ndarray) -> np.ndarray:  # Z‑Y‑X
    beta = np.arcsin(-R[..., 2, 0])
    alpha = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    gamma = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.rad2deg(np.stack([alpha, beta, gamma], axis=-1))


def _euler_xyz(R: np.ndarray) -> np.ndarray:  # X‑Y‑Z
    beta = np.arcsin(-R[..., 0, 2])
    alpha = np.arctan2(R[..., 1, 2], R[..., 2, 2])
    gamma = np.arctan2(R[..., 0, 1], R[..., 0, 0])
    return np.rad2deg(np.stack([alpha, beta, gamma], axis=-1))


###############################################################################
# ISBJointAngleCalculator (unchanged interface, internals refactored)          #
###############################################################################

class ISBJointAngleCalculator:
    def __init__(self, marker_data: Dict[str, np.ndarray], side: str = "R"):
        self.m = marker_data
        self.side = side.upper()
        assert self.side in {"R", "L"}
        self.nf = next(iter(marker_data.values())).shape[0]

    # ---------------- pelvis --------------------------------------------------

    def _pelvis_cs(self):
        la, ra = self.m["Lt_ASIS"], self.m["Rt_ASIS"]
        lps, rps = self.m["Lt_PSIS"], self.m["Rt_PSIS"]
        origin = (la + ra) / 2
        z = _unit(ra - la)
        ps_mid = (lps + rps) / 2
        x_tmp = ps_mid - origin
        x = _unit(np.cross(x_tmp, z))
        y = _unit(np.cross(z, x))
        R = _rot_from_axes(x, y, z)
        return origin, R

    # --------------- hip centres (Davis) -------------------------------------
    def _hip_joint_centres(self, O, R):
        la, ra = self.m["Lt_ASIS"], self.m["Rt_ASIS"]
        width = np.linalg.norm(ra - la, axis=1)
        C = np.array([-0.24, 0.30, 0.33]) # Per ISB, Davis et al.
        offs = np.array([-9.9, -10.9, 7.3]) # Per ISB, Davis et al.

        # Per ISB, le componenti x,y,z sono definite rispetto al CS pelvico
        # x: anteriore (+), y: superiore (+), z: destra (+)
        # Quindi per il centro dell'anca destra (rispetto all'origine del CS pelvico):
        # x_offset = C[0] * width + offs[0]  (negativo per la dx secondo Davis, ma il paper ISB non lo specifica come negativo a priori, lo gestisce con vR/vL)
        # y_offset = C[1] * width + offs[1]  (negativo)
        # z_offset = C[2] * width + offs[2]  (negativo per la dx)
        # L'implementazione originale usa:
        # vR = np.stack([-v[:, 0], -v[:, 1], -v[:, 2]], axis=-1)
        # vL = np.stack([v[:, 0], -v[:, 1], -v[:, 2]], axis=-1)
        # Questo implica che v[:,0] è l'offset laterale (asse Z pelvico), v[:,1] l'offset A/P (asse X pelvico), v[:,2] l'offset verticale (asse Y pelvico)
        # Adattiamo i coefficienti C e offs all'ordine atteso dal codice:
        # C_adj = [C[2], C[0], C[1]] # z, x, y del paper per v[0], v[1], v[2]
        # offs_adj = [offs[2], offs[0], offs[1]]
        # v[:,0] (laterale) = C[2]*width + offs[2]
        # v[:,1] (A/P)     = C[0]*width + offs[0]
        # v[:,2] (verticale)= C[1]*width + offs[1]
        
        # L'implementazione fornita usa vR e vL con segni specifici.
        # Mantengo la logica originale per coerenza con il codice fornito.
        v = np.empty((self.nf, 3))
        for i in range(3):
            v[:, i] = C[i] * width + offs[i] # Questi v[0],v[1],v[2] sono usati per costruire vR e vL
                                            # v[0] influenza l'asse X di vR/vL (destra/sinistra nel CS pelvico)
                                            # v[1] influenza l'asse Y di vR/vL (anteriore/posteriore nel CS pelvico)
                                            # v[2] influenza l'asse Z di vR/vL (superiore/inferiore nel CS pelvico)
        
        # Segni come da codice originale (per CS Pelvico X-ant, Y-sup, Z-dx):
        # HJC_R = O_pelvis + R_pelvis @ [ -v[0], -v[1], -v[2] ]^T
        # HJC_L = O_pelvis + R_pelvis @ [  v[0], -v[1], -v[2] ]^T
        # vR è il vettore offset nel CS pelvico per l'anca destra
        # vL è il vettore offset nel CS pelvico per l'anca sinistra

        vR_local = np.stack([-v[:, 0], -v[:, 1], -v[:, 2]], axis=-1) # Offset per anca destra nel CS pelvico
        vL_local = np.stack([ v[:, 0], -v[:, 1], -v[:, 2]], axis=-1) # Offset per anca sinistra nel CS pelvico

        # Trasformazione in CS globale
        # R è (nf, 3, 3), Rt è (nf, 3, 3)
        # vR_local[..., None] è (nf, 3, 1)
        # (Rt @ vR_local[..., None]) è (nf, 3, 1), .squeeze(-1) lo fa diventare (nf, 3)
        Rt = np.transpose(R, (0, 2, 1)) # Da CS locale a globale
        hjcR = O + (Rt @ vR_local[..., None]).squeeze(-1)
        hjcL = O + (Rt @ vL_local[..., None]).squeeze(-1)
        return hjcL, hjcR

    # --------------- segment CS helpers --------------------------------------
    def _thigh_cs(self, hip):
        side_prefix = "Rt" if self.side == "R" else "Lt"
        lat, med = self.m[f"{side_prefix}_Femoral_Lateral_Epicn"], self.m[f"{side_prefix}_Femoral_Medial_Epicn"]
        knee = (lat + med) / 2
        y = _unit(hip - knee)
        z_tmp = lat - med # Vettore da epicondilo mediale a laterale
        x = _unit(np.cross(y, z_tmp)) # Asse X anteriore (flessione positiva attorno a Z)
        z = _unit(np.cross(x,y)) # Asse Z laterale (destra per anca dx, sinistra per anca sx)

        # L'asse Z del CS della coscia ISB punta medialmente per la coscia destra
        # e lateralmente per la coscia sinistra.
        # Il codice fornito fa:
        # z = _unit(lat - med) # da mediale a laterale
        # if self.side == "L": z = -z # quindi per sinistra punta da laterale a mediale
        # Questo significa che Z punta sempre verso l'epicondilo laterale se self.side == 'R'
        # e verso l'epicondilo mediale se self.side == 'L'.
        # Tuttavia, le convenzioni ISB per la coscia:
        # Y: asse longitudinale, da centro ginocchio a centro anca
        # Z: asse di flesso-estensione, punta lateralmente per la coscia destra, medialmente per sinistra
        # X: anteriore, X = Y x Z
        # Manteniamo il codice fornito:
        z_from_code = _unit(lat - med)
        if self.side == "L":
            z_from_code = -z_from_code
        # x_from_code = _unit(np.cross(y, z_from_code))
        # R_from_code = _rot_from_axes(x_from_code, y, z_from_code)
        # Controlliamo la coerenza con ISB (Grood & Suntay per ginocchio X-Y-Z)
        # Per il ginocchio, l'asse di flessione/estensione è l'asse Z della coscia.
        # Una rotazione positiva attorno a Z_coscia è flessione.
        # Z_coscia (asse flottante) punta lateralmente per la coscia destra.
        
        # Riproponendo con la convenzione ISB per la coscia (Z anatomico laterale):
        # Y_thigh = _unit(hip - knee) (da ginocchio ad anca)
        # Z_thigh_anatomical = _unit(lat - med) # da mediale a laterale
        # if self.side == "L": Z_thigh_anatomical = -Z_thigh_anatomical # Z laterale per entrambi
        # X_thigh_anatomical = _unit(np.cross(Y_thigh, Z_thigh_anatomical)) # X anteriore
        # R_thigh_isb = _rot_from_axes(X_thigh_anatomical, Y_thigh, Z_thigh_anatomical)
        # Confrontando con il codice fornito:
        # y_code = y (OK)
        # z_code = z_from_code (punta lateralmente per coscia DX, medialmente per coscia SX)
        # x_code = _unit(np.cross(y_code, z_code))
        # Sembra che z_code sia l'asse anatomico Z per il ginocchio ISB (asse di flessione)
        x = _unit(np.cross(y, z_from_code))
        R = _rot_from_axes(x, y, z_from_code)
        return knee, R


    def _shank_cs(self, knee):
        side_prefix = "Rt" if self.side == "R" else "Lt"
        lat, med = self.m[f"{side_prefix}_Lateral_Malleolus"], self.m[f"{side_prefix}_Medial_Malleolus"]
        ankle = (lat + med) / 2
        y = _unit(knee - ankle) # Da caviglia a ginocchio
        # Convenzione ISB Tibia (simile a coscia):
        # Y: da centro caviglia a centro ginocchio
        # Z: asse di flesso-estensione, punta lateralmente per tibia destra, medialmente per sinistra
        # X: anteriore, X = Y x Z
        z_from_code = _unit(lat - med) # Da mediale a laterale
        if self.side == "L":
            z_from_code = -z_from_code # Punta lateralmente per DX, medialmente per SX
        x = _unit(np.cross(y, z_from_code))
        R = _rot_from_axes(x, y, z_from_code)
        return ankle, R

    def _foot_cs(self, ankle): # ankle è l'origine del CS del piede
        side_prefix = "Rt" if self.side == "R" else "Lt"
        heel = self.m[f"{side_prefix}_Calcaneous_Post"] # CAL
        # mt5 = self.m[f"{side_prefix}_Metatarsal_Phal_V"] # DPM (distal phalanx metatarsal V) o MTH5 (metatarsal head V)
        # mt1 = self.m[f"{side_prefix}_Metatarsal_Phal_I"] # DPM1 o MTH1
        # ISB usa PVM e PDM (proximal V e I metatarsal) o marker su testa metatarsale
        # Il codice fornito usa Metatarsal_Phal_V e Metatarsal_Phal_I
        # Assumiamo che questi siano i marker sulla testa del V e I metatarso (MTH5, MTH1)
        mt5 = self.m[f"{side_prefix}_Metatarsal_Phal_V"]
        mt1 = self.m[f"{side_prefix}_Metatarsal_Phal_I"]

        # Asse X del piede ISB: da calcagno a punto medio tra teste metatarsali (o marker dell'avampiede)
        # x_progressione = _unit(((mt1 + mt5) / 2) - heel) # Asse di progressione del piede
        
        # L'implementazione originale usa:
        # x = _unit(((mt1 + mt5) / 2) - heel)
        # y = _unit(np.cross(x, (mt5 - mt1)))
        # z = _unit(np.cross(x, y))
        # if self.side == "L": z = -z # Questo z è l'asse di flessione/estensione della caviglia
        # Questo è l'asse y anatomico (verticale del piede), non l'asse y del CS della tibia
        # CS Piede ISB (Wu 2002):
        # Yp: da VMH (V metatarsal head) a CAL (calcaneus) o asse longitudinale. Il codice usa il contrario.
        # Xp: Yp x (VMH - DMH) (DMH è II metatarsal head, o I metatarsal head), punta dorsalmente.
        # Zp: asse di flesso-estensione, punta lateralmente per piede destro. Zp = Xp x Yp
        # Il codice fornito:
        # x_foot = _unit(((mt1 + mt5) / 2) - heel) # da tallone a centro meta
        # temp_vec_meta = mt5 - mt1 # da meta1 a meta5
        # y_foot = _unit(np.cross(x_foot, temp_vec_meta)) # y_foot punta superiormente (dorsalmente)
        # z_foot = _unit(np.cross(x_foot, y_foot)) # z_foot punta lateralmente (per piede destro)
        # if self.side == "L": z_foot = -z_foot # z_foot punta medialmente (per piede sinistro)
        # Questo z_foot è l'asse di rotazione per flessione plantare/dorsale
        # Manteniamo la logica del codice fornito:
        x = _unit(((mt1 + mt5) / 2) - heel)
        y_axis_tmp = mt5 - mt1 # Vettore da I a V metatarso (mediale -> laterale per piede destro)
        y = _unit(np.cross(x, y_axis_tmp)) # y punta superiormente (dorsalmente)
        z = _unit(np.cross(x, y)) # z punta lateralmente per piede destro (asse di flessione)
        if self.side == "L": # Per piede sinistro, mt5-mt1 è laterale->mediale. cross(x, lat->med) -> y dorsale. cross(x,y) -> z mediale.
            z = -z # z punta lateralmente anche per piede sinistro se invertito.
                   # Tuttavia il codice originale NON inverte z per il piede sinistro qui,
                   # lo fa dopo R = _rot_from_axes(x,y,z) se side=="L". Questo è strano.
                   # Ricontrollando il codice originale:
                   # z = _unit(np.cross(x, y))
                   # if self.side == "L": z = -z # Questo è corretto per avere Z sempre laterale
                   # R = _rot_from_axes(x, y, z) # Usa questo Z corretto.

        # L'implementazione originale aveva un potenziale bug se z non era corretto prima di _rot_from_axes
        # ma poi lo correggeva per il calcolo degli angoli di eulero.
        # La versione nel prompt ha:
        # z = _unit(np.cross(x, y))
        # if self.side == "L": z = -z # <--- Questa riga è stata aggiunta/corretta nel prompt rispetto ad alcune versioni
        R = _rot_from_axes(x, y, z)
        return ankle, R


    def _knee_flex_simple(self, thigh_y: np.ndarray, shank_y: np.ndarray) -> float:
        thigh_y = np.asarray(thigh_y)
        shank_y = np.asarray(shank_y)
        if thigh_y.ndim == 2:
            dot = np.einsum("ij,ij->i", thigh_y, shank_y)
        else:
            dot = np.dot(thigh_y, shank_y)
        dot = np.clip(dot, -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def compute(self) -> pd.DataFrame:
        ang = {k: [] for k in [
            "hip_flexext", "hip_abdad", "hip_intext",
            "knee_flexext", "knee_abdad", "knee_intext",
            "ankle_pfdf", "ankle_inv_ev", "ankle_intext"]}

        Pelvis_O, Pelvis_R = self._pelvis_cs()
        hjcL, hjcR = self._hip_joint_centres(Pelvis_O, Pelvis_R)
        hip_jc = hjcR if self.side == "R" else hjcL

        # Pre-calcola tutti i CS per tutti i frame (vettorizzato)
        Thigh_O_all = np.empty_like(hip_jc)
        Thigh_R_all = np.empty_like(Pelvis_R)
        Shank_O_all = np.empty_like(hip_jc)
        Shank_R_all = np.empty_like(Pelvis_R)
        Foot_O_all = np.empty_like(hip_jc) # L'origine del piede è la caviglia
        Foot_R_all = np.empty_like(Pelvis_R)

        Thigh_O_all, Thigh_R_all = self._thigh_cs(hip_jc)
        Shank_O_all, Shank_R_all = self._shank_cs(Thigh_O_all) # Thigh_O_all è il knee_center
        Foot_O_all, Foot_R_all = self._foot_cs(Shank_O_all) # Shank_O_all è l'ankle_center
        
        # Calcoli angolari vettorizzati
        # Anca: ZYX (flessione, abd/add, rotazione int/ext)
        R_hp_all = np.transpose(Pelvis_R, (0, 2, 1)) @ Thigh_R_all # Pelvis_R.T @ Thigh_R
        hip_e_all = _euler_zyx(R_hp_all)
        ang["hip_flexext"] = hip_e_all[:, 0]
        ang["hip_abdad"] = hip_e_all[:, 1]
        ang["hip_intext"] = hip_e_all[:, 2]

        # Ginocchio:
        # Flessione semplice (angolo tra assi Y)
        thigh_y_all = Thigh_R_all[:, :, 1] # (nf, 3)
        shank_y_all = Shank_R_all[:, :, 1] # (nf, 3)
        ang["knee_flexext"] = self._knee_flex_simple(thigh_y_all, shank_y_all)
        
        # Abd/Add, Rotazione interna/esterna con XYZ di Grood e Suntay
        # R_kn = Thigh_R.T @ Shank_R
        R_kn_all = np.transpose(Thigh_R_all, (0, 2, 1)) @ Shank_R_all
        kn_e_all = _euler_xyz(R_kn_all) # X (abd/add), Y (rot int/ext), Z (fless/est)
                                        # Nota: il codice originale usa kn_e[1] per abd/add e kn_e[2] per rotazione
                                        # Questo implica che l'ordine di Euler per il ginocchio è YXZ o XZY
                                        # Se _euler_xyz è X-Y-Z, allora:
                                        # alpha (attorno a X_prox) -> abd/add
                                        # beta (attorno a Y_floated) -> fless/est
                                        # gamma (attorno a Z_dist) -> rot int/ext
                                        # Il codice originale associa:
                                        # kn_e[0] (alpha) a nulla di esplicito (ma sarebbe abd/add se usasse G&S X-Y-Z)
                                        # kn_e[1] (beta) a knee_abdad -> Questo è sbagliato per G&S XYZ, beta è flessione
                                        # kn_e[2] (gamma) a knee_intext -> Questo è rotazione interna/esterna (OK)

                                        # ISB raccomanda Grood & Suntay (1983) per ginocchio e caviglia.
                                        # Per il ginocchio, la sequenza di rotazione è flessione-estensione, seguita da
                                        # abduzione-adduzione, seguita da rotazione interna-esterna.
                                        # Z (asse di flessione, nel segmento prossimale, coscia)
                                        # X (asse di abd/add, nel segmento distale, tibia)
                                        # Y (asse di rotazione, asse lungo del segmento distale, tibia)
                                        # Questo è un sistema di assi non ortogonale e richiede una costruzione specifica.
                                        # L'uso di decomposizione di Eulero XYZ su R_Thigh.T @ R_Shank
                                        # è una semplificazione comune ma non strettamente G&S.
                                        # Se _euler_xyz(R) calcola angoli X-Y-Z (alpha, beta, gamma)
                                        # R = Rz(gamma)Ry(beta)Rx(alpha)
                                        # R_kn = R_thigh.T @ R_shank
                                        # Se gli assi dei CS sono allineati con X-avanti, Y-superiore, Z-destra
                                        # per la coscia (prossimale) e la tibia (distale):
                                        # rotazione attorno a X_coscia -> abd/add
                                        # rotazione attorno a Y_coscia_ruotato -> rotazione interna/esterna
                                        # rotazione attorno a Z_coscia_doppiamente_ruotato -> flessione/estensione
                                        # Il codice originale mappa:
                                        # ang["knee_abdad"].append(kn_e[1])
                                        # ang["knee_intext"].append(kn_e[2])
                                        # Questo implica che per _euler_xyz(R) -> (a,b,g)
                                        # b è abd/add, g è rotazione
                                        # Questo è coerente con una sequenza ZYX se R è R_distal.T @ R_proximal,
                                        # oppure XYZ se R è R_proximal.T @ R_distal e gli assi sono scambiati.
                                        # Data la definizione di _euler_xyz e l'uso, sembra che beta sia
                                        # l'angolo di "abduzione" e gamma quello di "rotazione".
        ang["knee_abdad"] = kn_e_all[:, 1] # Angolo beta da _euler_xyz
        ang["knee_intext"] = kn_e_all[:, 2] # Angolo gamma da _euler_xyz


        # Caviglia: XYZ (flessione plantare/dorsale, inversione/eversione, rotazione int/ext)
        # R_an = Shank_R.T @ Foot_R
        R_an_all = np.transpose(Shank_R_all, (0, 2, 1)) @ Foot_R_all
        an_e_all = _euler_xyz(R_an_all) # X (dorsiflex/plantarflex), Y (inversion/eversion), Z (rot int/ext)
        ang["ankle_pfdf"] = an_e_all[:, 0]
        ang["ankle_inv_ev"] = an_e_all[:, 1]
        ang["ankle_intext"] = an_e_all[:, 2]

        return pd.DataFrame(ang)

# --- Fine del codice ISB ---

def masked_angle_loss_isb(pred_coords, target_coords, mask, markers_names, angle_loss_type="l1"):
    """
    Calcola la loss sugli angoli articolari ISB ignorando i frame mascherati.
    Args:
        pred_coords (torch.Tensor): Posizioni articolari predette (B, F, J, 3) in metri.
        target_coords (torch.Tensor): Posizioni articolari ground truth (B, F, J, 3) in metri.
        mask (torch.Tensor): Maschera dei frame validi (B, F, J, 1) o (B, F).
        markers_names (List[str]): Lista dei nomi dei marker, nell'ordine della dimensione J.
        side (str): Lato per il calcolo degli angoli ("R" o "L"). Default "R".
        angle_loss_type (str): Tipo di loss sugli angoli ("l1" o "l2"). Default "l1".

    Returns:
        torch.Tensor: Mean Absolute Error o Mean Squared Error sugli angoli (in gradi) sui frame validi.
    """
    batch_size = pred_coords.shape[0]
    num_frames = pred_coords.shape[1]
    device = pred_coords.device
    dtype = pred_coords.dtype

    # Prepara la maschera a livello di frame (B, F)
    if mask.ndim == 4 and mask.shape[3] == 1: # (B, F, J, 1)
        # Usa la maschera del primo marker come rappresentativa per il frame
        # o mask.all(dim=2).squeeze(-1) se tutti i marker devono essere validi
        # o mask.any(dim=2).squeeze(-1) se almeno un marker deve essere valido
        frame_mask = mask[..., 0, 0].bool()
    elif mask.ndim == 3: # (B, F, J) - NUOVA CONDIZIONE
        # Similmente, puoi scegliere come aggregare la dimensione J
        # Qui usiamo la maschera del primo marker
        frame_mask = mask[..., 0].bool()
    elif mask.ndim == 2: # (B, F)
        frame_mask = mask.bool()
    else:
        raise ValueError(f"Forma della maschera non supportata: {mask.shape}. Gestite: (B,F,J,1), (B,F,J), (B,F).")


    total_angle_error = torch.tensor(0.0, device=device, dtype=dtype)
    total_valid_angle_components = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Nomi degli angoli come prodotti da ISBJointAngleCalculator.compute()
    # L'ordine è importante se si accede per indice, ma useremo i nomi delle colonne.
    # angle_names = ["hip_flexext", "hip_abdad", "hip_intext",
    #                "knee_flexext", "knee_abdad", "knee_intext",
    #                "ankle_pfdf", "ankle_inv_ev", "ankle_intext"]
    
    num_angle_types = 0

    for b in range(batch_size):
        pred_sample_np = pred_coords[b].detach().cpu().numpy()   # (F, J, 3)  
        target_sample_np = target_coords[b].detach().cpu().numpy() # (F, J, 3)
        
        active_frames_indices = torch.where(frame_mask[b])[0]
        if len(active_frames_indices) == 0:
            continue

        # Filtra solo i frame attivi per l'input a ISB (ottimizzazione)
        pred_marker_data_active = {
            name: pred_sample_np[active_frames_indices.cpu().numpy(), i, :]
            for i, name in enumerate(markers_names)
        }
        target_marker_data_active = {
            name: target_sample_np[active_frames_indices.cpu().numpy(), i, :]
            for i, name in enumerate(markers_names)
        }
        
        error_side = 0
        
        for side in ["R", "L"]:

            try:
                pred_calculator = ISBJointAngleCalculator(pred_marker_data_active, side=side)
                pred_angles_df = pred_calculator.compute() # DataFrame (active_F, num_angles)
                
                target_calculator = ISBJointAngleCalculator(target_marker_data_active, side=side)
                target_angles_df = target_calculator.compute() # DataFrame (active_F, num_angles)
            except Exception as e:
                # print(f"Warning: ISB computation failed for sample {b}, side {side}. Error: {e}")
                # Potresti voler registrare l'errore o gestire diversamente
                # In questo caso, saltiamo il campione per la loss degli angoli se il calcolo fallisce
                # o assegnare una loss penalizzante elevata. Per ora, lo saltiamo.
                # Per debug:
                # print("Pred marker data keys:", pred_marker_data_active.keys())
                # print("Pred marker data shapes:", {k: v.shape for k,v in pred_marker_data_active.items()})
                # print("Target marker data keys:", target_marker_data_active.keys())
                # print("Target marker data shapes:", {k: v.shape for k,v in target_marker_data_active.items()})
                # if hasattr(pred_calculator, 'm'):
                #     print("Markers in pred_calculator:", pred_calculator.m.keys())
                # if hasattr(target_calculator, 'm'):
                #     print("Markers in target_calculator:", target_calculator.m.keys())
                continue


            if num_angle_types == 0: # Imposta una volta
                num_angle_types = pred_angles_df.shape[1]
                if num_angle_types == 0 : # Nessun angolo calcolato
                    continue


            pred_angles_tensor = torch.tensor(pred_angles_df.values, device=device, dtype=dtype) # (active_F, num_angles)
            target_angles_tensor = torch.tensor(target_angles_df.values, device=device, dtype=dtype) # (active_F, num_angles)

            if angle_loss_type == "l1":
                diff_angles = torch.abs(pred_angles_tensor - target_angles_tensor)
            elif angle_loss_type == "l2":
                diff_angles = torch.square(pred_angles_tensor - target_angles_tensor)
            else:
                raise ValueError(f"Tipo di loss per angoli non supportato: {angle_loss_type}")

            total_angle_error += diff_angles.sum()
            total_valid_angle_components += pred_angles_tensor.numel() # num_active_frames_in_sample * num_angle_types

        if total_valid_angle_components == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        error_side += (total_angle_error / total_valid_angle_components)
        print(error_side)
    
    return error_side


def masked_loss_velocity_robust(predicted, target, mask, delta=1.0):
    """
    Robust masked velocity loss using Huber loss (SmoothL1) for outlier resistance.
    
    Why use Huber loss instead of MSE:
    - MSE (L2) is sensitive to outliers/jitter in motion data
    - Huber combines L2 for small errors (smooth gradients) with L1 for large errors (outlier resistance)
    - Delta parameter controls the transition point between L2 and L1 behavior
    - More stable gradients lead to better training convergence
    
    Uses central differences (t-1, t+1) instead of (t, t+1) for more robust velocity estimation:
    - Reduces sensitivity to single-frame noise
    - Smoother velocity estimates by averaging forward/backward differences
    
    Args:
        predicted: Tensor of shape (batch_size, seq_len, num_joints, 3)
        target: Tensor of shape (batch_size, seq_len, num_joints, 3)
        mask: Tensor of shape (batch_size, seq_len, num_joints) or (batch_size, seq_len)
        delta: Huber loss delta parameter (default=1.0mm). Transition point between L2 and L1.
    
    Returns:
        Scalar tensor with the robust masked velocity error in mm/s
    """
    assert predicted.shape == target.shape
    
    # Handle different mask shapes
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1).expand(-1, -1, predicted.shape[2])
    
    assert mask.shape == predicted.shape[:-1]
    
    if predicted.shape[1] <= 2:  # Need at least 3 frames for central differences
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    
    # Central differences: v(t) = (x(t+1) - x(t-1)) / 2
    # More robust than forward differences
    velocity_predicted = (predicted[:, 2:] - predicted[:, :-2]) / 2.0
    velocity_target = (target[:, 2:] - target[:, :-2]) / 2.0
    
    # Mask requires all three frames (t-1, t, t+1) to be valid
    velocity_mask = mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]
    
    # Huber loss (SmoothL1) for outlier resistance
    # delta controls transition between L2 (quadratic) and L1 (linear)
    velocity_diff = velocity_predicted - velocity_target
    velocity_error = torch.norm(velocity_diff, dim=-1)
    
    # Apply Huber loss: 0.5*x^2 if |x| < delta, else delta*(|x| - 0.5*delta)
    huber_loss = torch.where(
        velocity_error < delta,
        0.5 * velocity_error ** 2,
        delta * (velocity_error - 0.5 * delta)
    )
    
    # Apply mask and compute mean
    masked_errors = huber_loss * velocity_mask
    mask_sum = torch.sum(velocity_mask) + 1e-8
    masked_mean_error = torch.sum(masked_errors) / mask_sum
    
    return masked_mean_error * 1000  # Convert to mm/s
