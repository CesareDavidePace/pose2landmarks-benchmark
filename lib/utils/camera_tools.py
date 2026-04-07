import json
import os

import numpy as np
import torch

def triangulate_points_batch(pts_2d: torch.Tensor, proj_mats: torch.Tensor) -> torch.Tensor:
    """
    Triangulate 3D points from multiple camera views using batched DLT.

    Args:
        pts_2d: Tensor of shape (B, C, J, 2) - 2D keypoints
        proj_mats: Tensor of shape (B, C, 3, 4) - projection matrices

    Returns:
        pts_3d: Tensor of shape (B, J, 3) - triangulated 3D keypoints
    """
    B, C, J, _ = pts_2d.shape
    device = pts_2d.device

    # Expand projection matrices to match joints: (B, C, J, 3, 4)
    proj_mats = proj_mats.unsqueeze(2).expand(-1, -1, J, -1, -1)

    # Prepare rows of A matrix for each camera and joint
    x = pts_2d[..., 0].unsqueeze(-1)  # (B, C, J, 1)
    y = pts_2d[..., 1].unsqueeze(-1)  # (B, C, J, 1)

    P0 = proj_mats[..., 0, :]  # (B, C, J, 4)
    P1 = proj_mats[..., 1, :]
    P2 = proj_mats[..., 2, :]

    A1 = x * P2 - P0  # (B, C, J, 4)
    A2 = y * P2 - P1

    A = torch.cat([A1, A2], dim=-2)  # (B, C*2, J, 4) → incorrect dim order
    A = A.permute(0, 2, 1, 3).reshape(B * J, 2 * C, 4)  # → (B*J, 2C, 4)

    # SVD to solve AX = 0
    _, _, V = torch.linalg.svd(A)  # V: (B*J, 4, 4)
    X = V[:, -1]  # (B*J, 4)

    # Normalize homogeneous coordinates
    X = X[:, :3] / X[:, 3:].clamp(min=1e-6)

    return X.view(B, J, 3)  # (B, J, 3)

def get_projection_matrices(self, camera_ids: list[int]) -> np.ndarray:
    """
    Returns projection matrices for the selected camera IDs.
    This should load from calibration files or predefined intrinsics/extrinsics.
    """
    projection_matrices = []
    json_path = os.environ.get("MOVE4D_CAMERA_JSON", "data/move4d/cams/camera_matrices.json")

    with open(json_path, "r") as f:
        camera_matrices = json.load(f)

    for cam_id in camera_ids:
        # Load calibration data for this camera
        camera_matrix = camera_matrices[cam_id]["proj_matrix"]
        projection_matrices.append(camera_matrix)
    return np.stack(projection_matrices)  # (num_cams, 3, 4)

def get_selected_camera_ids(self, subject: str) -> list[int]:
    """Returns the camera IDs to use for this subject."""
    if self.selected_cameras is None:
        return None  # Default: use all cameras

    # Just flatten the list of groups
    cam_ids = [cam_id for group in self.selected_cameras.values() for cam_id in group]
    return sorted(list(set(cam_ids)))
    
def get_selected_camera_groups(camera_config, method, num_cameras):
    """
    Loads camera groups from a YAML file given a method and number of cameras.

    Args:
        yaml_path (str): Path to the YAML file.
        method (str): Method name, e.g., "evenly_spaced".
        num_cameras (int): Number of cameras, e.g., 2.

    Returns:
        list: List of selected camera groups.
    """

    try:
        groups = camera_config["camera_experiments"][method][num_cameras]["groups"]
        return groups
    except KeyError as e:
        raise ValueError(f"Could not find method '{method}' with {num_cameras} cameras in the YAML file.") from e

def triangulate_points(points_2d, projection_matrices):
    """
    Triangulate 3D points from multiple 2D views using Direct Linear Transform (DLT).
    
    Args:
        points_2d: List of 2D points from each camera [num_cameras, num_joints, 2]
        projection_matrices: Camera projection matrices [num_cameras, 3, 4]
        
    Returns:
        points_3d: Triangulated 3D points [num_joints, 3]
    """
    num_cameras, num_joints = points_2d.shape[:2]
    points_3d = np.zeros((num_joints, 3))
    
    for j in range(num_joints):
        # Build the A matrix for DLT
        A = np.zeros((num_cameras * 2, 4))
        
        for i in range(num_cameras):
            x, y = points_2d[i, j]
            P = projection_matrices[i]
            
            A[i*2] = x * P[2] - P[0]
            A[i*2+1] = y * P[2] - P[1]
        
        # Solve using SVD
        _, _, vh = np.linalg.svd(A)
        point_3d_homogeneous = vh[-1]
        
        # Convert from homogeneous to Euclidean coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
        points_3d[j] = point_3d
    
    return points_3d

def triangulate_points_torch(points_2d, projection_matrices, confidences=None):
    """
    Triangulate 3D points from multiple 2D views using DLT with PyTorch.
    
    Args:
        points_2d: Tensor of 2D points from each camera [num_cameras, num_joints, 2]
        projection_matrices: Camera projection matrices [num_cameras, 3, 4]
        confidences: Optional confidence scores [num_cameras, num_joints]
        
    Returns:
        points_3d: Triangulated 3D points [num_joints, 3]
    """
    num_cameras, num_joints = points_2d.shape[:2]
    points_3d = torch.zeros((num_joints, 3), device=points_2d.device)
    
    for j in range(num_joints):
        # Create weighted A matrix for DLT
        A = torch.zeros((num_cameras * 2, 4), device=points_2d.device)
        
        for i in range(num_cameras):
            x, y = points_2d[i, j]
            P = projection_matrices[i]
            
            # Weight equations by confidence if provided
            weight = 1.0
            if confidences is not None:
                weight = confidences[i, j]
                if weight < 0.1:  # Skip low confidence detections
                    continue
            
            A[i*2] = weight * (x * P[2] - P[0])
            A[i*2+1] = weight * (y * P[2] - P[1])
        
        # Solve using SVD
        try:
            _, _, vh = torch.linalg.svd(A)
            point_3d_homogeneous = vh[-1]
            
            # Convert from homogeneous to Euclidean coordinates
            point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
            points_3d[j] = point_3d
        except:
            # Fallback if SVD fails
            points_3d[j] = torch.zeros(3, device=points_2d.device)
    
    return points_3d

def project_3d_to_2d(points_3d, projection_matrix):
    """
    Project 3D points to 2D using camera projection matrix.
    
    Args:
        points_3d: 3D points [num_points, 3]
        projection_matrix: Camera projection matrix [3, 4]
        
    Returns:
        points_2d: Projected 2D points [num_points, 2]
    """
    # Convert to homogeneous coordinates
    if isinstance(points_3d, np.ndarray):
        num_points = points_3d.shape[0]
        homogeneous_points = np.hstack((points_3d, np.ones((num_points, 1))))
        # Project
        projected_points = projection_matrix @ homogeneous_points.T
        # Normalize
        points_2d = projected_points[:2] / projected_points[2]
        return points_2d.T
    else:  # Torch tensor
        num_points = points_3d.shape[0]
        homogeneous_points = torch.cat((points_3d, torch.ones((num_points, 1), device=points_3d.device)), dim=1)
        # Project
        projected_points = projection_matrix @ homogeneous_points.T
        # Normalize
        points_2d = projected_points[:2] / projected_points[2]
        return points_2d.T

def reprojection_error(points_3d, points_2d, projection_matrices):
    """
    Calculate reprojection error for triangulated points.
    
    Args:
        points_3d: 3D points [num_points, 3]
        points_2d: 2D points from each camera [num_cameras, num_points, 2]
        projection_matrices: Camera projection matrices [num_cameras, 3, 4]
        
    Returns:
        error: Mean reprojection error
    """
    num_cameras, num_points = points_2d.shape[:2]
    total_error = 0.0
    
    for i in range(num_cameras):
        # Project 3D points to camera view
        projected_points = project_3d_to_2d(points_3d, projection_matrices[i])
        
        # Calculate L2 distance between original and reprojected points
        if isinstance(points_2d, np.ndarray):
            error = np.linalg.norm(points_2d[i] - projected_points, axis=1).mean()
        else:  # Torch tensor
            error = torch.norm(points_2d[i] - projected_points, dim=1).mean()
        
        total_error += error
    
    return total_error / num_cameras


def test_triangulation_numpy():
    # Create synthetic 3D points
    num_joints = 5
    points_3d_gt = np.random.rand(num_joints, 3) * 10  # Ground truth 3D points

    # Define two simple camera projection matrices (camera 1 and camera 2)
    P1 = np.array([[1000, 0, 320, 0],
                   [0, 1000, 240, 0],
                   [0, 0, 1, 0]])
    
    P2 = np.array([[1000, 0, 320, -100],
                   [0, 1000, 240, 0],
                   [0, 0, 1, 0]])

    projection_matrices = np.stack([P1, P2])

    # Project 3D points to 2D
    points_2d = np.array([project_3d_to_2d(points_3d_gt, P) for P in projection_matrices])

    # Triangulate 3D points
    points_3d_reconstructed = triangulate_points(points_2d, projection_matrices)

    # Compute reprojection error
    error = reprojection_error(points_3d_reconstructed, points_2d, projection_matrices)
    print("NumPy Test - Reprojection Error:", error)
    print("Ground Truth 3D:\n", points_3d_gt)
    print("Reconstructed 3D:\n", points_3d_reconstructed)


    # try torch based triangulation
    points_2d_torch = torch.tensor(points_2d, dtype=torch.float32)
    projection_matrices_torch = torch.tensor(projection_matrices, dtype=torch.float32)
    points_3d_reconstructed_torch = triangulate_points_torch(points_2d_torch, projection_matrices_torch)
    error_torch = reprojection_error(points_3d_reconstructed_torch, points_2d_torch, projection_matrices_torch)
    print("Torch Test - Reprojection Error:", error_torch)
    print("Ground Truth 3D:\n", points_3d_gt)
