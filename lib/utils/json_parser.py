import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def load_json(json_path: str) -> Dict:
    """
    Load and parse a JSON file containing camera and keypoint data.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        Dict: The loaded JSON data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_kp2d_json(self, file_path: str) -> np.ndarray:
    """
    Load 2D keypoints from KP2D.json.
    
    Assumes format:
    {
        "camera_id_1": [[x1, y1, conf1], [x2, y2, conf2], ...],  # per frame
        "camera_id_2": ...
    }

    Returns:
        (num_frames, num_joints, 2) array for the first available camera.
    """
    with open(file_path, "r") as f:
        kp2d_data = json.load(f)

    # Pick the first available camera if no camera subset is specified
    selected_camera_ids = self.get_selected_camera_ids("")

    if selected_camera_ids:
        available = [str(cid) for cid in selected_camera_ids if str(cid) in kp2d_data]
    else:
        available = list(kp2d_data.keys())

    if not available:
        raise ValueError(f"No usable cameras found in {file_path}!")

    first_cam = available[0]
    frames = kp2d_data[first_cam]

    # Drop confidence (assumes [x, y, conf])
    arr = np.array(frames, dtype=np.float32)[..., :2]  # (T, J, 2)
    
    return arr
    
def extract_camera_info(json_data: Dict) -> Dict[int, Dict]:
    """
    Extract camera information including projection matrices from JSON data.
    
    Args:
        json_data (Dict): The loaded JSON data
        
    Returns:
        Dict[int, Dict]: Dictionary mapping camera IDs to their projection matrices
    """
    cameras = {}
    
    # Group annotations by camera
    for annotation in json_data.get('annotations', []):
        camera_id = annotation.get('camera')
        
        if camera_id not in cameras:
            # Get projection matrix
            proj_matrix = np.array(annotation.get('proj_matrix', []))
            rows = annotation.get('proj_matrix_rows', 3)
            cols = annotation.get('proj_matrix_cols', 4)
            
            # Reshape projection matrix properly
            if len(proj_matrix) == rows * cols:
                proj_matrix = proj_matrix.reshape(rows, cols)
            
            cameras[camera_id] = {
                'proj_matrix': proj_matrix,
                'frames': []
            }
        
        # Store frame data for this camera
        cameras[camera_id]['frames'].append({
            'frame': annotation.get('frame'),
            'keypoints': annotation.get('keypoints', []),
            'keypoints_scores': annotation.get('keypoints_scores', [])
        })
    
    return cameras

def extract_keypoints_2d(json_data: Dict, min_confidence: float = 0.3) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Extract 2D keypoints from all cameras, organized by frame and camera.
    
    Args:
        json_data (Dict): The loaded JSON data
        min_confidence (float): Minimum confidence score to consider a keypoint valid
        
    Returns:
        Dict[int, Dict[int, np.ndarray]]: Dictionary mapping frames to cameras to keypoints
    """
    fps = json_data.get('fps', 30)  # Get FPS from JSON or default to 90
    num_keypoints = None
    keypoints_by_frame = {}
    
    # Group annotations by frame and camera
    for annotation in json_data.get('annotations', []):
        frame = annotation.get('frame')
        camera_id = annotation.get('camera')
        keypoints = annotation.get('keypoints', [])
        scores = annotation.get('keypoints_scores', [])
        
        # Determine number of keypoints from first annotation
        if num_keypoints is None and keypoints:
            # Keypoints are typically stored as [x1,y1,x2,y2,...] flat list
            num_keypoints = len(keypoints) // 2
        
        # Initialize frame entry if not exists
        if frame not in keypoints_by_frame:
            keypoints_by_frame[frame] = {}
        
        # Reshape keypoints into (num_keypoints, 2) array
        keypoints_reshaped = np.array(keypoints).reshape(-1, 2)
        
        # Filter keypoints based on confidence score
        valid_keypoints = keypoints_reshaped.copy()
        for i, score in enumerate(scores):
            if score < min_confidence:
                valid_keypoints[i] = np.array([-1, -1])  # Mark as invalid
        
        keypoints_by_frame[frame][camera_id] = valid_keypoints
    
    return keypoints_by_frame, fps

def organize_sequence_data(keypoints_by_frame: Dict[int, Dict[int, np.ndarray]], 
                           cameras: Dict[int, Dict], 
                           n_frames: int = 243) -> Tuple[Dict[int, np.ndarray], Dict[int, List[np.ndarray]], int]:
    """
    Organize sequence data into a format suitable for triangulation.
    
    Args:
        keypoints_by_frame (Dict): Dictionary mapping frames to cameras to keypoints
        cameras (Dict): Dictionary mapping camera IDs to projection matrices
        n_frames (int): Total number of frames to consider
        
    Returns:
        Tuple containing:
            - Dict mapping frames to camera-wise keypoints (frame -> cams -> kps)
            - Dict mapping frames to projection matrices (frame -> list of P matrices)
            - Number of detected cameras
    """
    organized_keypoints = {}
    organized_projection_matrices = {}
    
    # Sort frames to ensure proper sequence
    sorted_frames = sorted(keypoints_by_frame.keys())
    available_cameras = list(cameras.keys())
    num_cameras = len(available_cameras)
    
    for frame in sorted_frames[:n_frames]:  # Limit to n_frames
        frame_keypoints = keypoints_by_frame[frame]
        
        # For each frame, organize keypoints by camera
        organized_keypoints[frame] = {}
        projection_matrices = []
        
        for camera_id in available_cameras:
            if camera_id in frame_keypoints:
                organized_keypoints[frame][camera_id] = frame_keypoints[camera_id]
                projection_matrices.append(cameras[camera_id]['proj_matrix'])
        
        organized_projection_matrices[frame] = projection_matrices
    
    return organized_keypoints, organized_projection_matrices, num_cameras