
from typing import List, Tuple

markers_names = [
    "Lt_ASIS", # 0
    "Rt_ASIS", # 1
    "Rt_Radial_Styloid",
    "Lt_Radial_Styloid", # 3
    "Suprasternale",
    "Rt_Radiale", # 5
    "Lt_Radiale",
    "Lt_Sphyrion", # 7
    "Rt_Sphyrion",
    "Lt_Metatarsal_Phal_I", # 9
    "Rt_Metatarsal_Phal_I",
    "Lt_Acromion", # 11
    "Rt_Acromion",
    "Substernale", # 13
    "Rt_Iliocristale",
    "Lt_Iliocristale", # 15
    "Rt_Digit_II",
    "Lt_Digit_II", #    17
    "Nuchale",
    "Rt_Axilla_Post", # 19
    "Lt_Axilla_Post",
    "Rt_Olecranon", # 21
    "Lt_Olecranon",
    "Lt_Calcaneous_Post", # 23
    "Rt_Calcaneous_Post",
    "Lt_Femoral_Lateral_Epicn", # 25
    "Rt_Femoral_Lateral_Epicn",
    "Rt_Femoral_Medial_Epicn", # 27
    "Lt_Femoral_Medial_Epicn",
    "Rt_Ulnar_Styloid", # 29
    "Lt_Ulnar_Styloid",
    "Rt_Medial_Malleolus", # 31 
    "Lt_Medial_Malleolus",
    "Lt_Humeral_Medial_Epicn", # 33
    "Rt_Humeral_Medial_Epicn",
    "Rt_Lateral_Malleolus", # 35
    "Lt_Lateral_Malleolus",
    "Rt_Trochanterion", # 37
    "Lt_Trochanterion",
    "Rt_Metatarsal_Phal_V",# 39
    "Lt_Metatarsal_Phal_V",
    "Rt_Axilla_Ant", # 41
    "Lt_Axilla_Ant",
    "Cervicale", # 43
    "Lt_Humeral_Lateral_Epicn",
    "Rt_Humeral_Lateral_Epicn", # 45
    "Rt_Gonion",
    "Lt_Gonion", # 47
    "Lt_Clavicale",
    "Rt_Clavicale", # 49
    "Rt_PSIS",
    "Lt_PSIS", # 51
    "10th_Rib_Midspin", 
    "SACR" # 53
]

markers_names_lower_limb = [
    "Lt_ASIS", "Lt_Calcaneous_Post", "Lt_Femoral_Lateral_Epicn", "Lt_Lateral_Malleolus",
    "Lt_Femoral_Medial_Epicn", "Lt_Medial_Malleolus", "Lt_Trochanterion", "Lt_Digit_II",
    "Lt_Metatarsal_Phal_V", "Lt_Metatarsal_Phal_I", 
    "Rt_ASIS", "Rt_Calcaneous_Post", "Rt_Femoral_Lateral_Epicn", "Rt_Lateral_Malleolus",
    "Rt_Femoral_Medial_Epicn", "Rt_Medial_Malleolus", "Rt_Trochanterion","Rt_Digit_II", 
    "Rt_Metatarsal_Phal_V", "Rt_Metatarsal_Phal_I"
]

# Anatomical segments of interest (tupla: (proximal, distal))
SEGMENTS: List[Tuple[int, int]] = [
    (0, 1),    # pelvis width
    (0, 51),   # Lt_ASIS – Lt_PSIS (pelvis depth)
    (1, 50),   # Rt_ASIS – Rt_PSIS
    (25, 38),  # Lt femur
    (26, 37),  # Rt femur  (uso Rt_Trochanterion)
    (38, 36),  # Lt tibia (trochanterion → lat malleolus)
    (37, 35),  # Rt tibia
    (36, 9),   # Lt foot (lat malleolus → MTP I)
    (35, 10),  # Rt foot
]
