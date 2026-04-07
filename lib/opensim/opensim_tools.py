import opensim as osim
osim.Logger.setLevelString('Warn')
import os
from pathlib import Path
import concurrent.futures
import shutil
from tqdm import tqdm
import time
import pandas as pd

# import argparse
import yaml
import argparse

# Optionally, specify a log file to write the messages
osim.Logger.removeFileSink()  # Remove any existing log file sinks
osim.Logger.addFileSink('opensim.log')  # Add a new log file sink


class OpensimTools:
    """
    A general class for OpenSim operations including scaling models and performing inverse kinematics
    for multiple subjects and actions.
    """
    
    def __init__(self, 
                 base_model_path, 
                 markerSet_path, 
                 output_root_folder, 
                 setup_scaling_path,
                 dataset_root_folder,
                 subjects_characteristics_file=None,
                 geometry_search_path=None,
                 verbose=False,
                 use_multiprocessing=True,
                 max_workers=None,
                 use_apose=False):
        """
        Initialize OpensimTools with paths and settings.
        
        Args:
            base_model_path: Path to the base OpenSim model
            markerSet_path: Path to the marker set XML file
            output_root_folder: Root folder for all outputs
            dataset_root_folder: Root folder containing subject data
            geometry_search_path: Path to geometry files (optional)
            verbose: If True, print detailed logs
            use_multiprocessing: If True, use multiprocessing for inverse kinematics
            max_workers: Maximum number of workers for multiprocessing (None = auto)
        """
        self.base_model_path = base_model_path
        self.markerSet_path = markerSet_path
        self.output_root_folder = output_root_folder
        self.setup_scaling_path = setup_scaling_path
        self.dataset_root_folder = dataset_root_folder
        self.geometry_search_path = geometry_search_path
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        self.subjects_characteristics_file = (
            subjects_characteristics_file
            or os.path.join(dataset_root_folder, "SUBJECTS_CHARACTERISTICS.csv")
        )
        self.subjects_characteristics = pd.read_csv(self.subjects_characteristics_file, delimiter=";")

        self.use_apose = use_apose
        print(f"Using A-POSE: {self.use_apose} for scaling")

        # Add geometry path to search paths if provided
        if self.geometry_search_path:
            osim.ModelVisualizer.addDirToGeometrySearchPaths(self.geometry_search_path)
            
        # Create output root folder if it doesn't exist
        os.makedirs(self.output_root_folder, exist_ok=True)
        
        self.log(f"OpenSim version: {osim.__version__}")
        self.log(f"OpensimTools initialized with base model: {os.path.basename(self.base_model_path)}")
    
    def log(self, message):
        """Log messages if verbose is enabled"""
        if self.verbose:
            print(message)

    def get_anthropometry(self,  subject):
        """
        Create a specific output folder for a subject.

        """

        #get the subject characteristics
        subject_characteristics = self.subjects_characteristics[self.subjects_characteristics["Subject_code"] == subject]
        mass = subject_characteristics["Weight (kg)"].values[0]
        height = subject_characteristics["Height (cm)"].values[0]*10

        age = subject_characteristics["Age (years)"].values[0]

        # convert in double
        mass = float(mass)
        height = float(height)
        age = float(age)

        return mass, height, age

        
    
    def create_output_folder(self, subject):
        """Create a versioned output folder for a subject"""
        subject_output = os.path.join(self.output_root_folder, subject)
        output_folder = subject_output
        self.log(f"Output folder created at {output_folder}")
        # Create GT and PRED folders
        os.makedirs(os.path.join(output_folder, "GT"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "PRED"), exist_ok=True)
        
        return output_folder

    def time_range_from_static(self, trc_file):
        marker_data = osim.MarkerData(trc_file)
        start_time = marker_data.getStartFrameTime()
        end_time = marker_data.getLastFrameTime()


        time_range = osim.ArrayDouble()

        if self.use_apose:
            time_range.set(0, start_time)
            time_range.set(1, end_time)
        else:
            # get range 0 and 0.1
            time_range.set(0, start_time)
            time_range.set(1, start_time + 0.2 if start_time + 0.1 < end_time else end_time)
        return time_range

    
    def scale_model(self, subject, subject_folder, output_folder, marker_file_path, output_suffix="scaled"):
        """
        Scale the model for a specific subject using a single marker file.
        This function is completely agnostic to the marker file type (gt, pred, etc.).
        
        Args:
            subject: Subject ID
            subject_folder: Path to the subject's data folder
            output_folder: Output folder for the scaled model
            marker_file_path: Path to the marker file to use for scaling (A-POSE file)
            output_suffix: Suffix to add to the output model file names (default: "scaled")
            
        Returns:
            Path to the scaled model with markers
        """
        self.log(f"Scaling model for subject {subject} using marker file: {marker_file_path}")
        
        # Paths for the model with custom marker set
        model_markerSet_path = os.path.join(output_folder, f"gait2354_simbody_with_custom_markerSet.osim")
        
        # Verify the marker file exists
        if not os.path.exists(marker_file_path):
            raise FileNotFoundError(f"Marker file not found: {marker_file_path}")

            
        # Load the model
        model = osim.Model(self.base_model_path)
        self.log(f"Model {model.getName()} loaded successfully")
        
        # Load the new marker set
        new_marker_set = osim.MarkerSet(self.markerSet_path)
        self.log(f"MarkerSet {new_marker_set.getName()} loaded successfully")
        
        # Update the model's marker set with the new markers
        model.updateMarkerSet(new_marker_set)
        self.log(f"Model's marker set updated successfully")
        
        # Save the updated model to an XML file
        model.printToXML(model_markerSet_path)
        self.log(f"Model with updated marker set saved at {model_markerSet_path}")
        
        # Get anthropometry data
        mass, height, age = self.get_anthropometry(subject)
        
        # Setup paths for the output models
        model_scaled_path = os.path.join(output_folder, f"{subject}_{output_suffix}.osim")
        model_scaled_markers_path = model_scaled_path.replace(".osim", "_markers.osim")
        
        # Create and configure the scale tool
        scale_tool = osim.ScaleTool(self.setup_scaling_path)
        scale_tool.setName(f"{subject}_ScaleTool")
        scale_tool.setSubjectMass(mass)
        scale_tool.setSubjectHeight(height)
        scale_tool.setSubjectAge(age)
        
        # Set the generic model maker to use the model with marker set
        scale_tool.getGenericModelMaker().setModelFileName(model_markerSet_path)
        
        # Configure the model scaler
        model_scaler = scale_tool.getModelScaler()
        model_scaler.setApply(True)
        model_scaler.setMarkerFileName(marker_file_path)

        model_scaler.setTimeRange(self.time_range_from_static(marker_file_path))

        model_scaler.setPreserveMassDist(True)
        model_scaler.setOutputModelFileName(model_scaled_path)
        model_scaler.setOutputScaleFileName(os.path.join(output_folder, f"{subject}_{output_suffix}_scaling_factor.xml"))
        
        # Run model scaler
        model_scaler.processModel(model, "", mass)
        
        # Configure the marker placer
        marker_placer = scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_placer.setTimeRange(self.time_range_from_static(marker_file_path))
        marker_placer.setStaticPoseFileName(marker_file_path)
        marker_placer.setOutputModelFileName(model_scaled_markers_path)
        marker_placer.setMaxMarkerMovement(-1)
        
        # Run marker placer
        scaled_model = osim.Model(model_scaled_path)
        marker_placer.processModel(scaled_model)
        
        # Save the final scaled model
        scaled_model.printToXML(model_scaled_path)
        
        # Save the scale tool configuration
        scale_tool.printToXML(os.path.join(output_folder, f"{subject}_{output_suffix}_scale_set.xml"))
        
        self.log(f"Scaled model saved at {model_scaled_markers_path}")
        scaled_model.printToXML(model_scaled_markers_path)
        
        return model_scaled_markers_path

    def run_inverse_kinematics(self, model_path, trc_file, output_folder, data_type):
        """
        Run inverse kinematics for a single TRC file.
        
        Args:
            model_path: Path to the scaled model
            trc_file: Path to the TRC file
            output_folder: Base output folder
            data_type: Either 'GT' or 'PRED'
            
        Returns:
            Path to the output motion file
        """
        try:
            # Create specific output folder
            ik_output_folder = os.path.join(output_folder, data_type, "inverse_kinematics")
            os.makedirs(ik_output_folder, exist_ok=True)
            
            self.log(f"Processing {trc_file}")
            
            # Load model and create IK tool
            model = osim.Model(model_path)

            ik_tool = osim.InverseKinematicsTool()
            ik_tool.setModel(model)
            
            # Set file names
            filename = Path(trc_file).stem
            ik_tool.setName(filename)
            ik_tool.setMarkerDataFileName(str(trc_file))
            
            output_mot_file = os.path.join(ik_output_folder, f"{filename}.mot")
            ik_tool.setOutputMotionFileName(output_mot_file)
            ik_tool.setResultsDir(ik_output_folder)
            
            # Set time range
            m = osim.MarkerData(str(trc_file))
            start = m.getStartFrameTime()
            end = m.getLastFrameTime() 
            ik_tool.setStartTime(start)
            ik_tool.setEndTime(end)
            
            # Create and run setup file
            setup_file = os.path.join(output_folder, f"{filename}_ik_setup.xml")
            ik_tool.printToXML(setup_file)
            ik_tool.run()
            
            self.log(f"Inverse kinematics completed for {trc_file}, output: {output_mot_file}")
            return output_mot_file
        except Exception as e:
            self.log(f"ERROR processing {trc_file}: {str(e)}")
            print(f"ERROR IK processing {trc_file}: {str(e)}")
            return None
    
    def process_subject(self, subject):

        """
        Process all files for a single subject.
        
        Args:
            subject: Subject ID
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.log(f"Processing subject: {subject}")
            
            # Find subject folder
            subject_folder = os.path.join(self.dataset_root_folder, subject)
            if not os.path.exists(subject_folder):
                self.log(f"Subject folder {subject_folder} not found. Skipping.")
                return {"subject": subject, "status": "error", "message": "Subject folder not found"}
            
            # Create output folder for this subject
            output_folder = self.create_output_folder(subject)

            scaled_model_paths = {}
            
            # Scale the models
            if self.use_apose:
                try:
                    a_pose_pred_path = subject_folder + "/" + subject + "_" + "A-POSE_pred.trc"
                    a_pose_gt_path = subject_folder + "/" + subject + "_" + "A-POSE_gt.trc"
                    print(f"A-POSE paths: {a_pose_pred_path}, {a_pose_gt_path}")
                    scaled_model_paths["PRED"] = self.scale_model(
                        subject, subject_folder, output_folder, a_pose_pred_path, "PRED"
                    )
                    scaled_model_paths["GT"] = self.scale_model(
                        subject, subject_folder, output_folder, a_pose_gt_path, "GT"
                    )
                except Exception as e:
                    self.log(f"Error scaling model for subject {subject}: {str(e)}")
                    return {"subject": subject, "status": "error", "message": f"Scaling error: {str(e)}"}
            
            # Find all GT and PRED TRC files
            gt_files = list(Path(subject_folder).glob("*_gt.trc"))
            pred_files = list(Path(subject_folder).glob("*_pred.trc"))
            
            # Remove A-POSE files from processing for IK
            gt_files = [f for f in gt_files if "A-POSE" not in f.name]
            pred_files = [f for f in pred_files if "A-POSE" not in f.name]
            
            
            self.log(f"Found {len(gt_files)} GT files and {len(pred_files)} PRED files for subject {subject}")
            
            results = {
                "subject": subject,
                "status": "success",
                "scaled_models": scaled_model_paths,
                "gt_processed": [],
                "pred_processed": [],
                "errors": []
            }
            
            # Process GT files with GT scaled model
            for trc_file in gt_files:
                try:
                    if not self.use_apose:  
                        # If not using A-POSE, use the GT file for scaling
                        scaled_model_paths["GT"] = self.scale_model(
                            subject, subject_folder, output_folder, str(trc_file), "GT"
                        )

                    output_file = self.run_inverse_kinematics(
                        scaled_model_paths["GT"], str(trc_file), output_folder, "GT"
                    )
                    if output_file:
                        results["gt_processed"].append(output_file)
                except Exception as e:
                    error_msg = f"Error processing GT file {trc_file.name}: {str(e)}"
                    self.log(error_msg)
                    results["errors"].append(error_msg)
            
            # Process PRED files with PRED scaled model
            for trc_file in pred_files:
                try:
                    if not self.use_apose:
                        # If not using A-POSE, use the PRED file for scaling
                        scaled_model_paths["PRED"] = self.scale_model(
                            subject, subject_folder, output_folder, str(trc_file), "PRED"
                        )

                    output_file = self.run_inverse_kinematics(
                        scaled_model_paths["PRED"], str(trc_file), output_folder, "PRED"
                    )
                    if output_file:
                        results["pred_processed"].append(output_file)
                except Exception as e:
                    error_msg = f"Error processing PRED file {trc_file.name}: {str(e)}"
                    self.log(error_msg)
                    results["errors"].append(error_msg)
            
            self.log(f"Subject {subject} processing completed.")
            return results
            
        except Exception as e:
            self.log(f"Error processing subject {subject}: {str(e)}")
            print(f"Error processing subject {subject}: {str(e)}")
            return {"subject": subject, "status": "error", "message": str(e)}

    def process_dataset(self):
        """
        Process the entire dataset.
        
        Returns:
            Dictionary with processing results for each subject
        """
        # Get all subject folders
        subjects = [d for d in os.listdir(self.dataset_root_folder) 
                    if os.path.isdir(os.path.join(self.dataset_root_folder, d))]
        
        self.log(f"Found {len(subjects)} subjects in dataset: {', '.join(subjects)}")
        results = {}
        
        # Process each subject (can be parallelized further if needed)
        for subject in tqdm(subjects, desc="Processing subjects"):
            results[subject] = self.process_subject(subject)
        
        return results


# Run the full dataset processing
def run(
    dataset_root_folder,
    base_model_path,
    markerSet_path,
    output_root_folder,
    geometry_path,
    setup_scaling_path,
    use_apose,
    subjects_characteristics_file=None,
):

    # start timer
    start = time.time() 

    # create output folder with version_ + iteration. start with the last iteration
    iteration = 1
    output_folder = output_root_folder
    while os.path.exists(output_folder):
        iteration += 1
        output_folder = os.path.join(output_root_folder, f"_{iteration}")
    
    # Create the OpenSim tools instance
    osim_tools = OpensimTools(
        base_model_path=base_model_path,
        markerSet_path=markerSet_path,
        output_root_folder=output_folder,
        setup_scaling_path=setup_scaling_path,
        dataset_root_folder=dataset_root_folder,
        subjects_characteristics_file=subjects_characteristics_file,
        geometry_search_path=geometry_path,
        verbose=False,
        use_multiprocessing=False,
        use_apose=use_apose,
    )
    
    # Process the entire dataset
    results = osim_tools.process_dataset()
    
    # Summarize results
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    
    print(f"\nProcessing complete!")
    print(f"Total subjects: {len(results)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

    # end timer
    end = time.time()

    # print time in minute 
    print(f"Execution time: {(end - start)/60} minutes")
    
    # Save results summary
    import json
    with open(os.path.join(output_folder, "processing_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return output_folder

def main():
    # get the arguments
    parser = argparse.ArgumentParser(description="Process the entire dataset")
    
    # base model path
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base OpenSim model")
    parser.add_argument("--markerSet_path", type=str, required=True, help="Path to the marker set XML file")
    parser.add_argument("--geometry_path", type=str, required=True, help="Path to the geometry files")
    parser.add_argument("--setup_scaling_path", type=str, required=True, help="Path to the setup scaling file")
    parser.add_argument("--output_root_folder", type=str, required=True, help="Root folder for all outputs")
    parser.add_argument("--dataset_root_folder", type=str, required=True, help="Root folder containing subject data")
    parser.add_argument("--subjects_characteristics_file", type=str, help="Optional CSV with subject anthropometrics")
    parser.add_argument("--use_apose", action="store_true", help="Use A-POSE files for scaling")

    args = parser.parse_args()

    # process the dataset
    output_path = run(
        dataset_root_folder=args.dataset_root_folder,
        base_model_path=args.base_model_path,
        markerSet_path=args.markerSet_path,
        setup_scaling_path=args.setup_scaling_path,
        output_root_folder=args.output_root_folder,
        geometry_path=args.geometry_path,
        use_apose=args.use_apose,
        subjects_characteristics_file=args.subjects_characteristics_file,
    )

    return output_path

    

if __name__ == "__main__":
    main()
