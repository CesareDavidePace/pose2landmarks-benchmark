import numpy as np
from scipy.signal import butter, filtfilt



# function to calculate RMSE given joints
def calculate_rmse(predicted, actual):
    """
    Calculate the Root Mean Square Error (RMSE) between predicted and actual joint positions,
    returning both the mean RMSE and its standard deviation over all joints and batches.

    Args:
        predicted (np.ndarray): Predicted joint positions of shape [B, N, D].
        actual (np.ndarray): Actual joint positions of shape [B, N, D].

    Returns:
        tuple: (mean_rmse, std_rmse) both as floats.
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    if predicted.shape != actual.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}")

    squared_errors = (predicted - actual) ** 2  # shape [B, N, D]
    per_point_mse = np.mean(squared_errors, axis=-1)  # shape [B, N] -> MSE per joint
    per_point_rmse = np.sqrt(per_point_mse)           # shape [B, N] -> RMSE per joint

    mean_rmse = np.mean(per_point_rmse) * 1000             # scalare in mm
    std_rmse = np.std(per_point_rmse) * 1000            # scalare in mm

    return mean_rmse, std_rmse

class BodyAngleCalculator:
    def __init__(self, filter_angles=False, cutoff_freq=5.0, sampling_rate=100):
        """
        Initialize the calculator with an optional filter.
        Source for ISB standards:
        - Wu et al. (2002) - Hip, Knee, Ankle: https://doi.org/10.1016/S0021-9290(01)00222-6
        - Wu et al. (2005) - Shoulder, Elbow, Wrist: https://doi.org/10.1016/j.jbiomech.2004.05.042
        """
        self.filter_angles = filter_angles
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
    
    def get_markers_for_frame(self, predictions, marker_names, t):
        """
        Returns a dictionary of marker_name -> np.array([x, y, z]) for frame t
        """
        # predictions[t] is shape [53, 3]
        frame_markers = {}
        for i, name in enumerate(marker_names):
            # Convert to a NumPy array if predictions is a torch.Tensor
            # If predictions is already a NumPy array, you can skip .cpu().numpy()
            coord_3d = predictions[t, i].cpu().numpy()  # shape [3]
            frame_markers[name] = coord_3d
        return frame_markers
            
    def calculate_error_dict(self, angles_gt, angles_pred):
        """
        Calculate the error between ground truth and predicted angles.
        :param angles_gt: Dictionary with ground truth angles.
        :param angles_pred: Dictionary with predicted angles.
        :return: Dictionary containing the error for each angle.
        """
        error_dict = {}
        for limb in angles_gt:
            error_dict[limb] = {}
            for key in angles_gt[limb]:
                error_dict[limb][key] = np.abs(angles_gt[limb][key] - angles_pred[limb][key])
        return error_dict

    def _vector_angle(self, v1, v2):
        """
        Compute the angle between two 3D vectors using the dot product formula.
        """
        v1, v2 = np.array(v1), np.array(v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if norm_v1 < 1e-10 or norm_v2 < 1e-10:
            return float('nan')  # Avoid division errors

        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180.0 / np.pi)  # Convert to degrees

    def _butter_lowpass_filter(self, data):
        """
        Apply a Butterworth low-pass filter to smooth the angles.
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def get_markers_for_frame(self, predictions, marker_names, t):
        """
        Returns a dictionary of marker_name -> np.array([x, y, z]) for frame t
        """
        # predictions[t] is shape [53, 3]
        frame_markers = {}
        for i, name in enumerate(marker_names):
            # Convert to a NumPy array if predictions is a torch.Tensor
            # If predictions is already a NumPy array, you can skip .cpu().numpy()
            coord_3d = predictions[t, i].cpu().numpy()  # shape [3]
            frame_markers[name] = coord_3d
        return frame_markers

    def estimate_HJC(self, markers):
        """
        Estimate Hip Joint Center (HJC) based on Harrington et al. (2007)
        DOI: https://doi.org/10.1016/j.jbiomech.2006.02.003
        """
        PW = np.linalg.norm(markers["Lt_ASIS"] - markers["Rt_ASIS"])
        PD = np.linalg.norm(markers["Lt_PSIS"] - markers["Rt_PSIS"])

        HJC_x = -0.24 * PD - 9.9
        HJC_y = -0.30 * PW - 10.9
        HJC_z = 0.33 * PW + 7.3

        SACR = (markers["Rt_PSIS"] + markers["Lt_PSIS"]) / 2
        return SACR + np.array([HJC_x, HJC_y, HJC_z])

    def compute_angles(self, markers):
        """
        Compute joint angles for Hip, Knee, and Ankle based on ISB standards.
        """
        angles = {"hip": {}, "knee": {}, "ankle": {}}

        # Hip Joint Center Estimation
        HJC = self.estimate_HJC(markers)

        # Hip Joint Angles
        v_thigh = HJC - markers["Rt_Femoral_Lateral_Epicn"]
        v_pelvis_vertical = markers["Suprasternale"] - markers["Substernale"]
        v_pelvis_frontal = markers["Lt_ASIS"] - markers["Rt_ASIS"]

        angles["hip"]["FE"] = self._vector_angle(v_thigh, v_pelvis_vertical)
        angles["hip"]["AB-AD"] = self._vector_angle(v_thigh, v_pelvis_frontal)
        angles["hip"]["ROT"] = self._vector_angle(v_thigh, markers["Rt_Femoral_Lateral_Epicn"] - markers["Rt_Femoral_Medial_Epicn"])

        # Knee Joint Angles
        v_shank = markers["Rt_Femoral_Lateral_Epicn"] - markers["Rt_Medial_Malleolus"]
        v_shank_longitudinal = markers["Rt_Femoral_Medial_Epicn"] - markers["Rt_Medial_Malleolus"]

        angles["knee"]["FE"] = self._vector_angle(v_thigh, v_shank)
        angles["knee"]["AB-AD"] = self._vector_angle(v_shank, v_pelvis_frontal)  # Knee varus/valgus
        angles["knee"]["ROT"] = self._vector_angle(v_shank_longitudinal, markers["Rt_Medial_Malleolus"] - markers["Rt_Lateral_Malleolus"])  # Tibial rotation

        # Ankle Joint Angles
        v_foot = markers["Rt_Medial_Malleolus"] - markers["Rt_Metatarsal_Phal_I"]
        v_shank_frontal = markers["Rt_Medial_Malleolus"] - markers["Rt_Lateral_Malleolus"]

        angles["ankle"]["FE"] = self._vector_angle(v_shank, v_foot)  # Dorsiflexion/Plantarflexion
        angles["ankle"]["AB-AD"] = self._vector_angle(v_shank_frontal, v_foot)  # Inversion/Eversion
        angles["ankle"]["ROT"] = self._vector_angle(v_foot, v_shank)  # Foot rotation

        if self.filter_angles:
            for joint in angles:
                for key in angles[joint]:
                    angles[joint][key] = self._butter_lowpass_filter([angles[joint][key]])[0]

        return angles


def _compute_marker_distance_errors(self):
    """
    Compute distance errors between predicted and ground truth markers for each sample.
    
    Returns:
    - Dictionary of distance errors organized by action and marker
    """
    distance_errors = {}
    
    # Iterate through raw predictions
    for pred_data in self.raw_predictions:
        subject_id = pred_data["subject_id"]
        action_id = pred_data["action_id"]
        prediction = np.array(pred_data["prediction"])
        ground_truth = np.array(pred_data["ground_truth"])
        
        # Ensure the marker names are in the same order as the data
        marker_names = markers_names  # from the imported markers_names
        
        # Compute distances for each frame
        for frame in range(prediction.shape[0]):
            # Compute Euclidean distance for each marker
            frame_distances = {}
            for i, marker_name in enumerate(marker_names):
                # Compute Euclidean distance between predicted and ground truth marker
                distance = np.linalg.norm(prediction[frame, i] - ground_truth[frame, i])
                
                # Organize by action and marker
                if action_id not in distance_errors:
                    distance_errors[action_id] = {}
                
                if marker_name not in distance_errors[action_id]:
                    distance_errors[action_id][marker_name] = []
                
                distance_errors[action_id][marker_name].append(distance)
    
    return distance_errors


# Example Usage
if __name__ == "__main__":
    example_markers = {
        "Rt_Femoral_Lateral_Epicn": np.array([0.8, 0.5, 0.3]),
        "Rt_Femoral_Medial_Epicn": np.array([0.85, 0.5, 0.35]),
        "Rt_Medial_Malleolus": np.array([1.0, 0.2, 0.1]),
        "Rt_Lateral_Malleolus": np.array([0.95, 0.2, 0.15]),
        "Rt_Metatarsal_Phal_I": np.array([1.2, 0.0, 0.0]),
        "Suprasternale": np.array([0.0, 2.5, 4.0]),
        "Substernale": np.array([0.0, 2.0, 3.5]),
        "Rt_ASIS": np.array([0.6, 1.5, 0.7]),
        "Lt_ASIS": np.array([0.5, 1.6, 0.8]),
        "Rt_PSIS": np.array([0.5, 1.0, 0.5]),
        "Lt_PSIS": np.array([0.4, 1.2, 0.6])
    }

    angle_calculator = BodyAngleCalculator(filter_angles=True)
    calculated_angles = angle_calculator.compute_angles(example_markers)
    print(calculated_angles)



