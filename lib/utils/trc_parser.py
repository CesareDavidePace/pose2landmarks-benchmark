import numpy as np
import re

def read_trc_file(file_path):
    """
    Read a .trc file and return the header info and data
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract header information
    header_info = {}
    header_info['path_file_type'] = lines[0].strip()

    data_info = lines[2].strip().split('\t')
    header_info['data_rate'] = float(data_info[0])
    header_info['camera_rate'] = float(data_info[1])
    header_info['num_frames'] = int(data_info[2])
    header_info['num_markers'] = int(data_info[3])
    header_info['units'] = data_info[4]

    # Get marker names
    marker_line = lines[3].strip().split('\t')
    marker_names = []
    for item in marker_line[2:]:
        if item and not re.match(r'X\d+|Y\d+|Z\d+', item):
            marker_names.append(item)

    # Read the data
    data_start_line = 5  # Data starts at line 7 (index 6)

    frames = np.zeros(header_info['num_frames'])
    times = np.zeros(header_info['num_frames'])
    markers_data = np.zeros((header_info['num_frames'], header_info['num_markers'], 3))

    for i in range(header_info['num_frames']):
        if i + data_start_line >= len(lines):
            print(f"Warning: Expected {header_info['num_frames']} frames but only found {i}")
            break

        line_data = lines[i + data_start_line].strip().split('\t')

        frames[i] = float(line_data[0])
        times[i] = float(line_data[1])

        for j in range(header_info['num_markers']):
            # X
            markers_data[i, j, 0] = float(line_data[2 + j * 3])  # X
            # Y
            markers_data[i, j, 1] = float(line_data[2 + j * 3 + 1])  # Y
            # Z
            markers_data[i, j, 2] = float(line_data[2 + j * 3 + 2])  # Z

            # X
            # markers_data[i, j, 0] = float(line_data[2 + j * 3 + 2]) # X = Z
            # Y
            # markers_data[i, j, 1] = float(line_data[2 + j * 3 + 1])  # Y = Y
            # Z
            # markers_data[i, j, 2] = - float(line_data[2 + j * 3])


    return header_info, marker_names, frames, times, markers_data

def create_trc_from_kp3d(data, kp3d_path, output_path, verbose=False):
    """
    Create a TRC file from KP3D data.
    
    Parameters:
    data (numpy.ndarray): Array of shape (frames, markers, 3) with XYZ coordinates
    kp3d_path (str): Path to the KP3D file
    output_path (str): Path to save the output TRC file
    """
    
    # read the kp3d file
    with open(kp3d_path, 'r') as f:
        lines = f.readlines()
        
    # extract the header
    header = lines[:6]
    
    # edit the path 
    header[0] = "PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_path)
    
    numbers = header[2].split()
    frame_rate = int(numbers[1])  # FrameRate
    
    # edit NumFrames 
    numbers[2] = str(data.shape[0])  # NumFrames
    numbers[-1] = str(data.shape[1])  # OrigNumFrames

    # write numbers back
    header[2] = "\t".join(numbers) + "\n"
    
    with open(output_path, 'w') as f:
        # Write header
        f.writelines(header)
        
        # Write data
        for frame_idx in range(data.shape[0]):
            time = frame_idx / frame_rate
            line = [str(frame_idx + 1), f"{time:.6f}"]
            for marker_idx in range(data.shape[1]):
                x, y, z = data[frame_idx, marker_idx]
                line.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                
            f.write("\t".join(line) + "\n")
            
    if verbose:
        print(f"TRC file created: {output_path}")

def load_trc(file_path: str) -> np.ndarray:
    """
    Parses a .trc file and returns a NumPy array of shape (num_frames, num_joints, 3).
    
    Args:
        file_path (str): Path to the .trc file.

    Returns:
        np.ndarray: 3D coordinates of shape (num_frames, num_joints, 3).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract metadata from third line
    metadata = lines[2].split()
    num_frames = int(metadata[2])  # NumFrames
    num_joints = int(metadata[3])  # NumMarkers

    # Extract joint names (for KP3D, they are in the fourth line)
    column_headers = lines[3].strip().split()
    joint_names = column_headers[2:]  # Skip "Frame#" and "Time"

    # Ensure the number of joints is consistent
    # assert len(joint_names) == num_joints * 3, f"Expected {num_joints*3} columns, got {len(joint_names)}"

    # Read numerical data
    data_lines = lines[6:]  # Skip header lines
    num_frames = len(data_lines)
    
    data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    for i, line in enumerate(data_lines):
        values = line.split()
        if len(values) < num_joints * 3 + 2:  # Skip incomplete lines
            continue
        joint_data = np.array(values[2:], dtype=np.float32).reshape(num_joints, 3)
        data[i] = joint_data

    return data

# Funzione per filtrare un array TRC dato l'insieme degli indici di frame validi
def filter_trc_by_frames(data: np.ndarray, frames: list) -> np.ndarray:
    return data[frames]
    
if __name__ == "__main__":
    raise SystemExit("Import this module from training/evaluation scripts instead of running it directly.")
