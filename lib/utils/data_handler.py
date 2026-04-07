import numpy as np
import json
import os
from lib.utils.trc_parser import load_trc, filter_trc_by_frames
import cv2
import json
import numpy as np
from scipy.linalg import rq  

def get_common_json_frames(camera_data: dict) -> set:
    """Restituisce l'insieme dei frame presenti in tutte le camere."""
    common_frames = None
    for cam in camera_data:
        frames = set(camera_data[cam]["frames"].keys())
        if common_frames is None:
            common_frames = frames
        else:
            common_frames &= frames
    return common_frames if common_frames is not None else set()

def get_valid_trc_frames(kp3d: np.ndarray, al: np.ndarray) -> list:
    """
    Restituisce la lista degli indici di frame in cui entrambi i file TRC (kp3d e al)
    non presentano valori mancanti (cioè -1).
    """
    n_frames = min(len(kp3d), len(al))
    valid_frames = []
    for i in range(n_frames):
        if np.any(kp3d[i] == -1) or np.any(al[i] == -1):
            continue
        valid_frames.append(i)
    return valid_frames

def get_common_valid_frames(kp3d: np.ndarray, al: np.ndarray, camera_data: dict) -> list:
    """
    Intersezione fra:
      1) frame validi dai TRC (senza -1)
      2) frame annotati in tutte le camere del JSON
    """
    valid_trc_frames = set(get_valid_trc_frames(kp3d, al))
    common_json_frames = get_common_json_frames(camera_data)
    common_frames = sorted(valid_trc_frames & common_json_frames)
    return common_frames

def build_aligned_kp2d(camera_data: dict, frames: list) -> dict:
    """
    Per ogni camera, restituisce due array:
      - keypoints:    (n_frames, n_joints, 3)
      - proj_matrices:(n_frames, R, C)
    dove R×C è la dimensione della projection matrix di quella camera.
    """
    aligned = {}
    for cam, cam_info in camera_data.items():
        kp_list = []
        P_list  = []
        for frame in frames:
            frame_dict = cam_info["frames"][frame]
            kp_list.append(frame_dict["keypoints"])
            P_list.append(frame_dict["proj_matrix"])
        aligned[cam] = {
            "keypoints":    np.stack(kp_list, axis=0),
            "proj_matrices": np.stack(P_list, axis=0),
            "K_orig": cam_info["K_orig"],
            "distCoeffs": cam_info["distCoeffs"],
        }
    return aligned

def load_and_align_data(kp3d_path: str, al_path: str, kp2d_path: str):
    """
    Carica TRC (kp3d, al), JSON 2D, trova i frame comuni validi,
    filtra i TRC e allinea i keypoints 2D + projection matrices.
    """
    # --- carica TRC 3D e AL
    kp3d = load_trc(kp3d_path)
    al   = load_trc(al_path)
    
    # --- carica JSON 2D
    camera_data, json_metadata = load_kp2d_json_dict(kp2d_path)

    # --- individua i frame utili
    common_frames = get_common_valid_frames(kp3d, al, camera_data)
    if not common_frames:
        raise ValueError("Non sono stati trovati frame comuni validi in tutte le sorgenti!")
    
    # --- filtra i TRC su quei frame
    kp3d_aligned = filter_trc_by_frames(kp3d, common_frames)
    al_aligned   = filter_trc_by_frames(al,   common_frames)
    
    # --- allinea i 2D e le projection matrices
    kp2d_aligned = build_aligned_kp2d(camera_data, common_frames)
    
    # --- prepara i nuovi metadata
    new_metadata = {
        "frames":      common_frames,
        "num_frames":  len(common_frames),
        "subject":     json_metadata["subject"],
        "movement":    json_metadata["movement"],
        "fps":         json_metadata["fps"],
        "num_joints":  json_metadata["num_joints"],
        "num_cameras": json_metadata["num_cameras"]
    }
    
    return kp3d_aligned, al_aligned, kp2d_aligned, new_metadata
    

# Funzione per caricare il JSON e organizzare le annotazioni per camera ed indicizzate per frame
def load_kp2d_json_dict(kp2d_path: str):
    with open(kp2d_path, 'r') as f:
        kp2d_data = json.load(f)

    camera_data = {}
    for ann in kp2d_data["annotations"]:
        cam   = ann["camera"]
        frame = ann["frame"]
        kps   = np.array(ann["keypoints_scores"], dtype=float).reshape(-1, 3)

        rows   = ann.get("proj_matrix_rows", 3)
        cols   = ann.get("proj_matrix_cols", 4)
        P      = np.array(ann["proj_matrix"], dtype=float, order="C").reshape(rows, cols)
        
        # -------- estrai intrinseca K dalla matrice di proiezione --------
        K_, R_ = rq(P[:3, :3])                 # RQ decomposition
        T = np.diag(np.sign(np.diag(K_)))      # assicura diag(K) > 0
        K_ = K_ @ T
        
        # nessuna distorsione dichiarata → vettore di zeri
        dist = np.zeros(5, dtype=float)
        # ----------------------------------------------------------
        #  initialise the per-camera record the first time we see it
        # ----------------------------------------------------------
        if cam not in camera_data:
            # extract the optical centre **once per camera**
            _, _, C_h, *_ = cv2.decomposeProjectionMatrix(P)
            centre = (C_h[:3] / C_h[3]).ravel()

            camera_data[cam] = dict(
                frames   = {},     # filled below
                centre   = centre, # (3,)
                P = P, # canonical 3×4 (all frames identical in this dataset)
                K_orig = K_,    # (3,3) camera intrinsics
                distCoeffs = dist,   # (5,) distortion coefficients
            )

        # store per-frame keypoints  (and P only if it actually varies)
        camera_data[cam]["frames"][frame] = dict(
            keypoints   = kps,
            proj_matrix = P,            # keep this if you need frame-wise Ps,
            P = P, # canonical 3×4 (all frames identical in this dataset)
            K_orig = K_,    # (3,3) camera intrinsics
            distCoeffs = dist,   # (5,) distortion coefficients
        )

    # ---- derive a few metadata counters for later convenience ----------
    sample_kps = next(iter(next(iter(camera_data.values()))["frames"].values()))
    num_joints = sample_kps["keypoints"].shape[0]

    metadata = dict(
        subject      = kp2d_data.get("subject", ""),
        movement     = kp2d_data.get("movement", ""),
        fps          = kp2d_data.get("fps", 0),
        num_joints   = num_joints,
        num_cameras  = len(camera_data)
    )
    return camera_data, metadata