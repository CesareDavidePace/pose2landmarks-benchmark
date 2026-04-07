import torch
from typing import Iterable, List, Optional, Tuple

from lib.utils.markers_names_move4d import markers_names as MOVE4D_MARKERS


AMASS_DSTFORMER_OUTPUT_MARKERS = [
    "RASIS_augmenter", "LASIS_augmenter", "RPSIS_augmenter", "LPSIS_augmenter",
    "RKnee_augmenter", "RMKnee_augmenter", "RAnkle_augmenter", "RMAnkle_augmenter",
    "RToe_augmenter", "R5meta_augmenter", "RCalc_augmenter",
    "LKnee_augmenter", "LMKnee_augmenter", "LAnkle_augmenter", "LMAnkle_augmenter",
    "LToe_augmenter", "LCalc_augmenter", "L5meta_augmenter",
    "RShoulder_augmenter", "LShoulder_augmenter", "C7_augmenter",
    "RElbow_augmenter", "RMElbow_augmenter", "RWrist_augmenter", "RMWrist_augmenter",
    "LElbow_augmenter", "LMElbow_augmenter", "LWrist_augmenter", "LMWrist_augmenter",
    "RThigh1_augmenter", "RThigh2_augmenter", "RThigh3_augmenter",
    "LThigh1_augmenter", "LThigh2_augmenter", "LThigh3_augmenter",
    "RSh1_augmenter", "RSh2_augmenter", "RSh3_augmenter",
    "LSh1_augmenter", "LSh2_augmenter", "LSh3_augmenter",
    "RHJC_augmenter", "LHJC_augmenter",
]

# OpenPose 20-joint input ordering used by AMASS openpose features.
OPENPOSE_20_MARKERS = [
    "Neck_openpose",
    "RShoulder_openpose",
    "LShoulder_openpose",
    "RHip_openpose",
    "LHip_openpose",
    "midHip_openpose",
    "RKnee_openpose",
    "LKnee_openpose",
    "RAnkle_openpose",
    "LAnkle_openpose",
    "RHeel_openpose",
    "LHeel_openpose",
    "RSmallToe_openpose",
    "LSmallToe_openpose",
    "RBigToe_openpose",
    "LBigToe_openpose",
    "RElbow_openpose",
    "LElbow_openpose",
    "RWrist_openpose",
    "LWrist_openpose",
]

def _normalize_marker_name(name: str) -> str:
    return name.replace("_augmenter", "").strip()


def _build_index_map(marker_names: List[str]) -> dict:
    return {_normalize_marker_name(name): idx for idx, name in enumerate(marker_names)}


def build_adjacency(num_nodes: int,
                    edges: Iterable[Tuple[int, int]],
                    add_self_loops: bool = True,
                    normalize: bool = True) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    if add_self_loops:
        adj.fill_diagonal_(1.0)

    if normalize:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    return adj


def _edges_from_names(marker_names: List[str],
                      edge_names: Iterable[Tuple[str, str]],
                      verbose: bool = False) -> List[Tuple[int, int]]:
    name_to_idx = _build_index_map(marker_names)
    edges = []
    missing = []
    for name_a, name_b in edge_names:
        idx_a = name_to_idx.get(_normalize_marker_name(name_a))
        idx_b = name_to_idx.get(_normalize_marker_name(name_b))
        if idx_a is None or idx_b is None:
            missing.append((name_a, name_b))
            continue
        edges.append((idx_a, idx_b))

    if verbose and missing:
        missing_preview = ", ".join([f"{a}-{b}" for a, b in missing[:5]])
        print(f"\033[93mMissing graph edges (showing up to 5): {missing_preview}\033[0m")

    return edges


def _move4d_edge_names() -> List[Tuple[str, str]]:
    return [
        ("Lt_ASIS", "Rt_ASIS"),
        ("Lt_PSIS", "Rt_PSIS"),
        ("Lt_ASIS", "Lt_PSIS"),
        ("Rt_ASIS", "Rt_PSIS"),
        ("Lt_ASIS", "Lt_Trochanterion"),
        ("Rt_ASIS", "Rt_Trochanterion"),
        ("Lt_Trochanterion", "Lt_Femoral_Lateral_Epicn"),
        ("Lt_Trochanterion", "Lt_Femoral_Medial_Epicn"),
        ("Rt_Trochanterion", "Rt_Femoral_Lateral_Epicn"),
        ("Rt_Trochanterion", "Rt_Femoral_Medial_Epicn"),
        ("Lt_Femoral_Lateral_Epicn", "Lt_Lateral_Malleolus"),
        ("Lt_Femoral_Medial_Epicn", "Lt_Medial_Malleolus"),
        ("Rt_Femoral_Lateral_Epicn", "Rt_Lateral_Malleolus"),
        ("Rt_Femoral_Medial_Epicn", "Rt_Medial_Malleolus"),
        ("Lt_Lateral_Malleolus", "Lt_Calcaneous_Post"),
        ("Lt_Lateral_Malleolus", "Lt_Metatarsal_Phal_I"),
        ("Lt_Lateral_Malleolus", "Lt_Metatarsal_Phal_V"),
        ("Rt_Lateral_Malleolus", "Rt_Calcaneous_Post"),
        ("Rt_Lateral_Malleolus", "Rt_Metatarsal_Phal_I"),
        ("Rt_Lateral_Malleolus", "Rt_Metatarsal_Phal_V"),
        ("Lt_Metatarsal_Phal_I", "Lt_Digit_II"),
        ("Rt_Metatarsal_Phal_I", "Rt_Digit_II"),
        ("Lt_Clavicale", "Cervicale"),
        ("Rt_Clavicale", "Cervicale"),
        ("Lt_Clavicale", "Lt_Acromion"),
        ("Rt_Clavicale", "Rt_Acromion"),
        ("Lt_Acromion", "Lt_Humeral_Lateral_Epicn"),
        ("Lt_Acromion", "Lt_Humeral_Medial_Epicn"),
        ("Rt_Acromion", "Rt_Humeral_Lateral_Epicn"),
        ("Rt_Acromion", "Rt_Humeral_Medial_Epicn"),
        ("Lt_Humeral_Lateral_Epicn", "Lt_Olecranon"),
        ("Rt_Humeral_Lateral_Epicn", "Rt_Olecranon"),
        ("Lt_Olecranon", "Lt_Ulnar_Styloid"),
        ("Rt_Olecranon", "Rt_Ulnar_Styloid"),
        ("Lt_Ulnar_Styloid", "Lt_Radial_Styloid"),
        ("Rt_Ulnar_Styloid", "Rt_Radial_Styloid"),
        ("Suprasternale", "Substernale"),
        ("Substernale", "10th_Rib_Midspin"),
        ("10th_Rib_Midspin", "SACR"),
        ("Nuchale", "Cervicale"),
    ]


def _amass_edge_names() -> List[Tuple[str, str]]:
    return [
        ("RASIS", "LASIS"),
        ("RPSIS", "LPSIS"),
        ("RASIS", "RPSIS"),
        ("LASIS", "LPSIS"),
        ("RASIS", "RHJC"),
        ("LASIS", "LHJC"),
        ("RHJC", "RThigh1"),
        ("RThigh1", "RThigh2"),
        ("RThigh2", "RThigh3"),
        ("RThigh3", "RKnee"),
        ("LHJC", "LThigh1"),
        ("LThigh1", "LThigh2"),
        ("LThigh2", "LThigh3"),
        ("LThigh3", "LKnee"),
        ("RKnee", "RMKnee"),
        ("LKnee", "LMKnee"),
        ("RKnee", "RSh1"),
        ("RSh1", "RSh2"),
        ("RSh2", "RSh3"),
        ("RSh3", "RAnkle"),
        ("LKnee", "LSh1"),
        ("LSh1", "LSh2"),
        ("LSh2", "LSh3"),
        ("LSh3", "LAnkle"),
        ("RAnkle", "RMAnkle"),
        ("LAnkle", "LMAnkle"),
        ("RAnkle", "RToe"),
        ("RAnkle", "R5meta"),
        ("RAnkle", "RCalc"),
        ("LAnkle", "LToe"),
        ("LAnkle", "L5meta"),
        ("LAnkle", "LCalc"),
        ("RShoulder", "C7"),
        ("LShoulder", "C7"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RMElbow"),
        ("RMElbow", "RWrist"),
        ("RWrist", "RMWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LMElbow"),
        ("LMElbow", "LWrist"),
        ("LWrist", "LMWrist"),
    ]


def _openpose_edge_names() -> List[Tuple[str, str]]:
    return [
        ("Neck_openpose", "RShoulder_openpose"),
        ("Neck_openpose", "LShoulder_openpose"),
        ("Neck_openpose", "midHip_openpose"),
        ("midHip_openpose", "RHip_openpose"),
        ("midHip_openpose", "LHip_openpose"),
        ("RHip_openpose", "RKnee_openpose"),
        ("RKnee_openpose", "RAnkle_openpose"),
        ("RAnkle_openpose", "RHeel_openpose"),
        ("RHeel_openpose", "RBigToe_openpose"),
        ("RHeel_openpose", "RSmallToe_openpose"),
        ("LHip_openpose", "LKnee_openpose"),
        ("LKnee_openpose", "LAnkle_openpose"),
        ("LAnkle_openpose", "LHeel_openpose"),
        ("LHeel_openpose", "LBigToe_openpose"),
        ("LHeel_openpose", "LSmallToe_openpose"),
        ("RShoulder_openpose", "RElbow_openpose"),
        ("RElbow_openpose", "RWrist_openpose"),
        ("LShoulder_openpose", "LElbow_openpose"),
        ("LElbow_openpose", "LWrist_openpose"),
        ("RHip_openpose", "LHip_openpose"),
    ]


def _is_openpose_marker_set(marker_names: List[str]) -> bool:
    return any("openpose" in name for name in marker_names)


def get_default_marker_names(dataset_type: str, num_joints: Optional[int] = None) -> List[str]:
    if dataset_type.lower() == "amass":
        if num_joints == len(OPENPOSE_20_MARKERS):
            return OPENPOSE_20_MARKERS
        return AMASS_DSTFORMER_OUTPUT_MARKERS
    return MOVE4D_MARKERS


def build_dataset_adjacency(dataset_type: str,
                            num_joints: int,
                            marker_names: List[str] | None = None,
                            verbose: bool = False) -> torch.Tensor:
    marker_names = marker_names or get_default_marker_names(dataset_type, num_joints)
    if len(marker_names) != num_joints:
        raise ValueError(
            f"Graph markers ({len(marker_names)}) do not match num_joints ({num_joints}) for {dataset_type}."
        )

    dataset = dataset_type.lower()
    if dataset == "amass":
        edge_names = _openpose_edge_names() if _is_openpose_marker_set(marker_names) else _amass_edge_names()
    else:
        edge_names = _move4d_edge_names()
    edges = _edges_from_names(marker_names, edge_names, verbose=verbose)
    return build_adjacency(num_joints, edges, add_self_loops=True, normalize=True)
