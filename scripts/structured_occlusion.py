#!/usr/bin/env python3
"""
Structured Occlusion Patterns for Robustness Testing
====================================================

Implements realistic occlusion patterns instead of random dropout:
1. Temporal occlusions: entire joints missing for consecutive frames
2. Spatial occlusions: entire body parts (e.g., right leg) missing
3. Camera-based occlusions: markers occluded based on viewing angle
4. Movement-specific occlusions: self-occlusions during specific poses

This is more realistic than random i.i.d. dropout per (joint, frame).
"""

import numpy as np
from typing import List, Tuple, Optional


class StructuredOcclusionPatterns:
    """
    Generate structured occlusion masks for motion capture data.
    
    All methods return a boolean mask of shape (T, J) where:
    - True = visible/keep
    - False = occluded/drop
    """
    
    def __init__(self, n_frames: int, n_joints: int, seed: int = 42):
        self.T = n_frames
        self.J = n_joints
        self.rng = np.random.RandomState(seed)
    
    def temporal_occlusion(
        self,
        joint_indices: List[int],
        duration_frames: int = 20,
        n_occlusions: int = 2
    ) -> np.ndarray:
        """
        Occlude specific joints for consecutive frames (simulates temporary tracking loss).
        
        Args:
            joint_indices: Which joints to occlude
            duration_frames: How many consecutive frames
            n_occlusions: Number of occlusion events
        
        Returns:
            Mask (T, J) with True=visible
        """
        mask = np.ones((self.T, self.J), dtype=bool)
        
        for _ in range(n_occlusions):
            # Pick random start frame
            start = self.rng.randint(0, max(1, self.T - duration_frames))
            end = min(start + duration_frames, self.T)
            
            # Occlude selected joints
            for j in joint_indices:
                if j < self.J:
                    mask[start:end, j] = False
        
        return mask
    
    def spatial_occlusion(
        self,
        body_part_indices: List[int],
        occlusion_rate: float = 0.5
    ) -> np.ndarray:
        """
        Occlude entire body parts (e.g., right leg) for random frames.
        
        Args:
            body_part_indices: Indices of joints in the body part
            occlusion_rate: Fraction of frames where body part is occluded
        
        Returns:
            Mask (T, J) with True=visible
        """
        mask = np.ones((self.T, self.J), dtype=bool)
        
        # Decide which frames have the occlusion
        occluded_frames = self.rng.rand(self.T) < occlusion_rate
        
        # Apply to all joints in body part
        for j in body_part_indices:
            if j < self.J:
                mask[occluded_frames, j] = False
        
        return mask
    
    def camera_view_occlusion(
        self,
        marker_positions: np.ndarray,
        camera_position: np.ndarray = np.array([0, 0, 3]),
        occlusion_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Simulate camera-based occlusions (markers behind body center).
        
        Args:
            marker_positions: (T, J, 3) marker coordinates
            camera_position: Camera location in 3D
            occlusion_threshold: Depth threshold for occlusion
        
        Returns:
            Mask (T, J) with True=visible
        """
        mask = np.ones((self.T, self.J), dtype=bool)
        
        # For each frame, compute marker depth relative to body center
        for t in range(self.T):
            body_center = np.mean(marker_positions[t], axis=0)
            
            for j in range(self.J):
                marker_pos = marker_positions[t, j]
                
                # If marker is behind body center relative to camera, occlude it
                to_camera = camera_position - body_center
                to_marker = marker_pos - body_center
                
                # Dot product: negative means marker is on opposite side
                depth = np.dot(to_marker, to_camera / (np.linalg.norm(to_camera) + 1e-6))
                
                if depth < -occlusion_threshold:
                    mask[t, j] = False
        
        return mask
    
    def self_occlusion_squat(
        self,
        knee_angle_deg: np.ndarray,
        threshold_deg: float = 90.0,
        occlude_lower_body: bool = True
    ) -> np.ndarray:
        """
        Simulate self-occlusion during deep squats (lower body hidden).
        
        Args:
            knee_angle_deg: (T,) array of knee flexion angles
            threshold_deg: Knee angle threshold for occlusion trigger
            occlude_lower_body: If True, occlude legs; otherwise occlude upper body
        
        Returns:
            Mask (T, J) with True=visible
        """
        mask = np.ones((self.T, self.J), dtype=bool)
        
        # Define lower body joints (approximate MOVE4D indices)
        lower_body = list(range(20, 35))  # Legs and feet (adjust based on marker set)
        upper_body = list(range(0, 20))   # Torso, arms, head
        
        # When knee angle < threshold (deep squat), occlude selected body part
        occluded_frames = knee_angle_deg < threshold_deg
        
        if occlude_lower_body:
            for j in lower_body:
                if j < self.J:
                    mask[occluded_frames, j] = False
        else:
            for j in upper_body:
                if j < self.J:
                    mask[occluded_frames, j] = False
        
        return mask
    
    def combined_realistic_occlusion(
        self,
        marker_positions: Optional[np.ndarray] = None,
        knee_angle_deg: Optional[np.ndarray] = None,
        severity: str = "moderate"
    ) -> np.ndarray:
        """
        Combine multiple occlusion types for realistic test scenario.
        
        Args:
            marker_positions: (T, J, 3) if available for camera occlusion
            knee_angle_deg: (T,) if available for self-occlusion
            severity: "mild", "moderate", or "severe"
        
        Returns:
            Mask (T, J) with True=visible
        """
        mask = np.ones((self.T, self.J), dtype=bool)
        
        # Configure severity
        if severity == "mild":
            n_temporal = 1
            duration = 10
            spatial_rate = 0.1
        elif severity == "severe":
            n_temporal = 4
            duration = 30
            spatial_rate = 0.4
        else:  # moderate
            n_temporal = 2
            duration = 20
            spatial_rate = 0.2
        
        # Apply temporal occlusions (random joints)
        random_joints = self.rng.choice(self.J, size=min(5, self.J), replace=False)
        temporal_mask = self.temporal_occlusion(
            joint_indices=random_joints.tolist(),
            duration_frames=duration,
            n_occlusions=n_temporal
        )
        mask = mask & temporal_mask
        
        # Apply spatial occlusion to right side (simulate single camera view)
        right_side = list(range(self.J // 2, self.J))  # Approximate right-side markers
        spatial_mask = self.spatial_occlusion(
            body_part_indices=right_side,
            occlusion_rate=spatial_rate
        )
        mask = mask & spatial_mask
        
        return mask


# === Example MOVE4D body part definitions ===
# These are approximate - adjust based on actual marker set

MOVE4D_BODY_PARTS = {
    "right_leg": [20, 21, 22, 23, 24, 25, 26, 27],  # Example indices
    "left_leg": [28, 29, 30, 31, 32, 33, 34, 35],
    "right_arm": [8, 9, 10, 11, 12],
    "left_arm": [13, 14, 15, 16, 17],
    "torso": [0, 1, 2, 3, 4, 5, 6, 7],
    "head": [36, 37, 38, 39],
}


def get_body_part_indices(part_name: str) -> List[int]:
    """Get joint indices for a named body part."""
    return MOVE4D_BODY_PARTS.get(part_name, [])


# === Usage Example ===
if __name__ == "__main__":
    # Example: create structured occlusions
    T, J = 243, 53  # MOVE4D dimensions
    
    occ = StructuredOcclusionPatterns(n_frames=T, n_joints=J, seed=42)
    
    # 1. Temporal occlusion (right leg disappears for 20 frames, twice)
    right_leg_indices = get_body_part_indices("right_leg")
    mask_temporal = occ.temporal_occlusion(
        joint_indices=right_leg_indices,
        duration_frames=20,
        n_occlusions=2
    )
    
    print(f"Temporal occlusion:")
    print(f"  Total frames: {T}")
    print(f"  Joints occluded: {len(right_leg_indices)}")
    print(f"  Visibility: {mask_temporal.sum() / mask_temporal.size * 100:.1f}%")
    
    # 2. Spatial occlusion (right arm missing in 30% of frames)
    right_arm_indices = get_body_part_indices("right_arm")
    mask_spatial = occ.spatial_occlusion(
        body_part_indices=right_arm_indices,
        occlusion_rate=0.3
    )
    
    print(f"\nSpatial occlusion:")
    print(f"  Body part: right arm ({len(right_arm_indices)} joints)")
    print(f"  Occlusion rate: 30%")
    print(f"  Visibility: {mask_spatial.sum() / mask_spatial.size * 100:.1f}%")
    
    # 3. Combined realistic scenario
    mask_combined = occ.combined_realistic_occlusion(severity="moderate")
    
    print(f"\nCombined realistic occlusion (moderate):")
    print(f"  Visibility: {mask_combined.sum() / mask_combined.size * 100:.1f}%")
    print(f"  Fully occluded frames: {(~mask_combined.any(axis=1)).sum()}/{T}")
