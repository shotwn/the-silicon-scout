import sys
import subprocess
import os

def _extract_tool_content(result: str) -> str:
    """Helper to parse <tool_result> tags if present."""
    result_token = "<tool_result>"
    result_end_token = "</tool_result>"
    if result_token in result and result_end_token in result:
        start_idx = result.index(result_token) + len(result_token)
        end_idx = result.index(result_end_token)
        return result[start_idx:end_idx].strip()
    return result

def fastjet_tool(
    input_file: str,
    numpy_read_chunk_size: int | None = None,
    size_per_row: int | None = None,
    output_dir: str | None = None,
    min_pt: float | None = None,
    no_label_input: bool = False,
):
    """
    Tool to run FastJet clustering on input data. 
    Uses the import_and_fastjet.py script. 
    If input data is blackbox or real life data, meaning it has no labels, set no_label_input to True. 
    If user says R&D data, meaning it has labels, leave no_label_input as False. 
    Stop generation and ask user if you are not sure.
    Will generate clustered output files in json format. 
    Files will be _background and _signal files if R&D data and only one _unlabeled file if unlabeled data.
    Args:
        input_file: Path to the input data file. Should be in .h5 format.
        numpy_read_chunk_size: Chunk size for reading numpy files.
        size_per_row: Size per row for processing. Default is 2100 for R&D data. Clarify based on data.
        output_dir: Directory to save output files.
        min_pt: Minimum pt in GeV threshold for clustering. Default is 1200.0.
        no_label_input: Whether the input file has labels. R&D data may have labels. Real data does not.
    Returns:
        A string summarizing the preprocessing results.
    """
    if input_file is None:
        return "Error: input_file parameter is required for FastJet tool."
    
    command = [
        f"{sys.executable}",
        "framework/tools/import_and_fastjet.py",
        "--input_file", input_file,
    ]

    if numpy_read_chunk_size is not None:
        command += ["--numpy_read_chunk_size", str(numpy_read_chunk_size)]
    if size_per_row is not None:
        command += ["--size_per_row", str(size_per_row)]
    if output_dir is not None:
        command += ["--output_dir", output_dir]
    if min_pt is not None:
        command += ["--min_pt", str(min_pt)]
    if no_label_input:
        command += ["--no_label_input"]

    try:
        print(f"Worker: Executing command: {' '.join(command)}")
        # Run the subprocess
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        # Process the output
        result_content = _extract_tool_content(result.stdout)
        return f"Data preprocessing completed successfully. Output:\n{result_content}"
        
    except subprocess.CalledProcessError as e:
        return f"Data preprocessing failed.\nError Code: {e.returncode}\nStderr: {e.stderr}"
    except Exception as e:
        return f"Unexpected execution error: {str(e)}"
    
def lacathode_preparation_tool(
    run_id: str,
    run_mode: str = 'training',
    input_background: str | None = None,
    input_signal: str | None = None,
    input_unlabeled: str | None = None,
    #output_dir: str | None = None,
    shuffle_seed: int | None = None,
    training_fraction: float | None = None,
    validation_fraction: float | None = None,
    side_band_min: float | None = None,
    min_mass: float | None = None,
    max_mass: float | None = None,
    side_band_max: float | None = None,
):
    """
    Tool to run LaCATHODE data preparation.
    Uses the lacathode_preparation.py script.
    Inputs are result of FastJet tool, fastjet tool result will give the file names.
    If fastjet tool result is _background and _signal files, use 'training' mode and provide both input_background and input_signal.
    If fastjet tool result is only _unlabeled file, use 'inference' mode and provide input_unlabeled.
    As a clarification, R&D data usually has both background and signal files, so use 'training' mode.
    Real life or blackbox data usually has only unlabeled file, so use 'inference' mode.
    Args:
        run_id: Unique identifier for the run. Determines the output directory. (required)
        run_mode: 'training' or 'inference'. Default is 'training'.
        input_background: Path to background data file (required for training mode).
        input_signal: Path to signal data file (required for training mode).
        input_unlabeled: Path to unlabeled data file (required for inference mode).
        shuffle_seed: Random seed for shuffling.
        training_fraction: Fraction of data to use for training.
        validation_fraction: Fraction of data to use for validation.
        side_band_min: Minimum mass for Sideband (SB) region in TeV.
        min_mass: Minimum mass for Signal Region (SR) window in TeV.
        max_mass: Maximum mass for Signal Region (SR) window in TeV.
        side_band_max: Maximum mass for Sideband (SB) region in TeV.
    """
    
    # Validation logic for required inputs based on mode
    if run_mode == 'training':
        if not input_background or not input_signal:
            return "Error: Training mode requires both input_background and input_signal arguments."
    elif run_mode == 'inference':
        if not input_unlabeled:
            return "Error: Inference mode requires input_unlabeled argument."
    else:
        return f"Error: Invalid run_mode '{run_mode}'. Must be 'training' or 'inference'."

    command = [
        f"{sys.executable}",
        "framework/tools/lacathode_preperation.py",
        "--run_mode", run_mode
    ]

    output_dir = f"toolout/lacathode_prepared_data/{run_id}/"

    # Add optional arguments if they exist
    if input_background:
        command += ["--input_background", input_background]
    if input_signal:
        command += ["--input_signal", input_signal]
    if input_unlabeled:
        command += ["--input_unlabeled", input_unlabeled]
    if output_dir:
        command += ["--output_dir", output_dir]
    if shuffle_seed is not None:
        command += ["--shuffle_seed", str(shuffle_seed)]
    if training_fraction is not None:
        command += ["--training_fraction", str(training_fraction)]
    if validation_fraction is not None:
        command += ["--validation_fraction", str(validation_fraction)]
    if side_band_min is not None:
        command += ["--side_band_min", str(side_band_min)]
    if min_mass is not None:
        command += ["--min_mass", str(min_mass)]
    if max_mass is not None:
        command += ["--max_mass", str(max_mass)]
    if side_band_max is not None:
        command += ["--side_band_max", str(side_band_max)]

    try:
        print(f"Worker: Executing command: {' '.join(command)}")
        # Run the subprocess
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        # Process the output
        result_content = _extract_tool_content(result.stdout)
        # Fallback if no tool_result tag, return full stdout
        if result_content == result.stdout:
            return f"LaCATHODE preparation completed.\n{result.stdout}"
        return f"LaCATHODE preparation completed successfully. Output:\n{result_content}"
        
    except subprocess.CalledProcessError as e:
        return f"LaCATHODE preparation failed.\nError Code: {e.returncode}\nStderr: {e.stderr}"
    except Exception as e:
        return f"Unexpected execution error: {str(e)}"

def lacathode_training_tool(
    run_id: str,
    data_dir: str,
    load_flow: bool = False,
    load_classifier: bool = False,
    epochs_flow: int | None = None,
    epochs_clf: int | None = None,
    plot: bool = False,
):
    """
    Tool to train the LaCATHODE model using the lacathode_trainer.py script.
    Inputs are result of LaCATHODE preparation tool.
    
    CRITICAL: This tool must be run AFTER 'lacathode_preparation_tool'.
    
    This process involves two main steps automatically handled by the script:
    1. Training a Normalizing Flow on background data (or loading a pre-trained one).
    2. Training a Classifier to distinguish real data from synthetic background in latent space.

    Oracle tool generally run after this tool to get final anomaly scores.
    
    Args:
        run_id: Unique identifier for the run. Determines the output directory. 
                You can use the same run_id as used in preparation step. Unless you make a new run. (required)
        data_dir: Directory containing prepared data from LaCATHODE preparation tool. (required)
        load_flow: If True, loads an existing Flow model from model_dir instead of retraining. 
                   Useful if the Flow is already good and you only want to retrain the classifier.
        load_classifier: If True, loads an existing Classifier instead of retraining.
        epochs_flow: Number of epochs for Flow training (default 100). 
                     Reduce for testing (e.g., 10), increase for better background modeling.
        epochs_clf: Number of epochs for Classifier training (default 50).
        plot: If True, generates an ROC curve plot after training.
    
    Returns:
        A string summarizing the training results, including ROC AUC if available.
    """
    
    command = [
        f"{sys.executable}",
        "framework/tools/lacathode_trainer.py",
    ]

    model_dir = f"toolout/lacathode_trained_models/{run_id}/"

    if data_dir:
        command += ["--data_dir", data_dir]
    if model_dir:
        command += ["--model_dir", model_dir]
    if load_flow:
        command += ["--load_flow"]
    if load_classifier:
        command += ["--load_classifier"]
    if epochs_flow is not None:
        command += ["--epochs_flow", str(epochs_flow)]
    if epochs_clf is not None:
        command += ["--epochs_clf", str(epochs_clf)]
    if plot:
        command += ["--plot"]

    try:
        print(f"Worker: Executing command: {' '.join(command)}")
        # Run the subprocess
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        # Process the output
        result_content = _extract_tool_content(result.stdout)
        
        # Fallback if no tool_result tag, return full stdout
        if result_content == result.stdout:
             return f"LaCATHODE training completed.\n{result.stdout}"
             
        return f"LaCATHODE training completed successfully. Output:\n{result_content}"
        
    except subprocess.CalledProcessError as e:
        return f"LaCATHODE training failed.\nError Code: {e.returncode}\nStderr: {e.stderr}"
    except Exception as e:
        return f"Unexpected execution error: {str(e)}"
    
def lacathode_oracle_tool(
    inference_file: str,
    data_dir: str,
    model_dir: str,
    run_id: str,
):
    """
    Tool to run the LaCATHODE Oracle Inference.
    Uses the lacathode_oracle.py script.
    
    This tool applies a trained LaCATHODE model to new data to generate anomaly scores.
    It requires access to the original training data directory to calibrate its scalers.

    model_dir is result of LaCATHODE training tool.
    inference_file is result of LaCATHODE preparation tool.
    
    Args:
        inference_file: Data file to predict on, this is result of preparation step. 
                        Let's say output of preparation step is in 'toolout/lacathode_prepared_data/{run_id}/' directory,
                        - e.g. 'toolout/lacathode_prepared_data/{run_id}/innerdata_combined.npy'.
                        This is a REQUIRED argument.
        data_dir: Directory containing the ORIGINAL training data (needed for calibration). 
                  Default is "./toolout/lacathode_prepared_data/{run_id}/".
        model_dir: Directory containing trained models. Also where output scores will be saved.
                   Default is "./toolout/lacathode_trained_models/{run_id}/".
        run_id: Unique identifier for the run. Determines the output directory. 
                You can use training run ID from the training step. (required)
    
    Returns:
        A string summarizing the inference results, including the path to the saved scores.
    """
    
    if not inference_file:
        return "Error: inference_file is a required argument for the Oracle tool."

    command = [
        f"{sys.executable}",
        "framework/tools/lacathode_oracle.py",
        "--inference_file", inference_file
    ]

    output_file = f"toolout/lacathode_trained_models/{run_id}/inference_scores.npy"

    if data_dir:
        command += ["--data_dir", data_dir]
    if model_dir:
        command += ["--model_dir", model_dir]
    if output_file:
        command += ["--output_file", output_file]

    try:
        print(f"Worker: Executing command: {' '.join(command)}")
        # Run the subprocess
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        # Process the output
        result_content = _extract_tool_content(result.stdout)
        
        # Fallback if no tool_result tag, return full stdout
        if result_content == result.stdout:
             return f"LaCATHODE Oracle inference completed.\n{result.stdout}"
             
        return f"LaCATHODE Oracle inference completed successfully. Output:\n{result_content}"
        
    except subprocess.CalledProcessError as e:
        return f"LaCATHODE Oracle inference failed.\nError Code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}"
    except Exception as e:
        return f"Unexpected execution error: {str(e)}"

def lacathode_report_generator_tool(
    data_file: str | None = None,
    scores_file: str | None = None,
    output_file: str | None = None,
    top_percentile: float | None = None,
    bin_count: int | None = None,
):
    """
    Tool to generate a human-readable anomaly report using lacathode_report_generator.py.

    Data file is result of LaCATHODE preparation tool.
    Scores file is result of LaCATHODE Oracle or Trainer tool.
    
    This tool analyzes the anomaly scores generated by the Oracle/Trainer, identifies 
    excess regions (hotspots) in the mass spectrum, and lists the top candidate events.
    
    Args:
        data_file: Path to the input data file (numpy .npy format). 
                   Usually 'toolout/lacathode_prepared_data/{run_id}/innerdata_combined.npy' (Required).
        scores_file: Path to the anomaly scores file (numpy .npy format).
                     Usually 'toolout/lacathode_trained_models/{run_id}/inference_scores.npy' (from Oracle) (Required).
        output_file: Path to save the generated report (default "llm_enhanced_report.txt").
        top_percentile: Percentile threshold to define anomaly candidates. Default 99.0 But you can experiment.
        bin_count: Number of bins for mass histogram. Default 18 But you can experiment.
    
    Returns:
        The content of the generated report.
    """
    
    if not data_file or not scores_file:
        return "Error: Both data_file and scores_file are required arguments."

    command = [
        f"{sys.executable}",
        "framework/tools/lacathode_report_generator.py",
        "--data_file", data_file,
        "--scores_file", scores_file,
    ]

    if output_file:
        command += ["--output_file", output_file]
    if top_percentile is not None:
        command += ["--top_percentile", str(top_percentile)]
    if bin_count is not None:
        command += ["--bin_count", str(bin_count)]

    try:
        print(f"Worker: Executing command: {' '.join(command)}")
        # Run the subprocess
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        # Process the output using the standard helper since tags are fixed
        result_content = _extract_tool_content(result.stdout)
        
        # Fallback if no tags found (though they should be there now)
        if result_content == result.stdout:
             return f"Report generated. Output:\n{result.stdout}"
             
        return f"Report generated successfully:\n\n{result_content}"
        
    except subprocess.CalledProcessError as e:
        return f"Report generation failed.\nError Code: {e.returncode}\nStderr: {e.stderr}"
    except Exception as e:
        return f"Unexpected execution error: {str(e)}"