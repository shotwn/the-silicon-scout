import sys
import subprocess
import os

from framework.tools.gemma_client import query_gemma_cloud

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
    If input data is blackbox or real life data, meaning it has no labels, set no_label_input=True. 
    If user says R&D data, meaning it has labels, no_label_input=False. 
    
    Files will be saved in output_dir.
    Expected output files:
    - For R&D data (with labels): {output_dir}/background_events.jsonl and {output_dir}/signal_events.jsonl
    - For blackbox/real data (no labels): {output_dir}/unlabeled_events.jsonl

    Expense is 10-20 CPU minutes for large datasets.
    Args:
        input_file: Path to the input data file. Should be in .h5 format.
        numpy_read_chunk_size: Chunk size for reading numpy files.
        size_per_row: Size per row for processing. Default is 2100 for R&D data. Clarify based on data.
        output_dir: Directory to save output files. Default is 'toolout/fastjet-output/'.
        min_pt: Minimum pt in GeV threshold for clustering. Default is 1200.0. Do not set too low. Do not mix with mass.
        no_label_input: Whether the input file has labels. R&D data may have labels. Real data does not.
    Returns:
        A string summarizing the preprocessing results.
    """
    if input_file is None:
        raise ValueError("Error: input_file parameter is required for FastJet tool.")
    
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
    return result.stdout

def propose_signal_regions_tool(
    input_background: str | None = None,
    input_signal: str | None = None,
    input_unlabeled: str | None = None,
    sigma_width: float = 4.0,
    resolution_ratio: float = 0.03,
    min_jet_pt: float = 1200.0,
    min_events_saturation: int = 40000
):
    """
    Analyzes input data files to recommend optimal mass windows for LaCATHODE.
    Prioritizes High Mass regions with sufficient statistics.

    Args:
        input_background: Path to background data file (JSONL or NPY). Used in Training Mode.
        input_signal: Path to signal data file (JSONL or NPY). Used in Training Mode.
        input_unlabeled: Path to unlabeled data file (JSONL or NPY). Used in Inference Mode.
        sigma_width: Width of the Signal Region (SR) in sigmas (default 4.0).
        resolution_ratio: Estimated detector mass resolution (default 0.03).
        min_jet_pt: The hardware trigger threshold used (default 1200.0 GeV). 
                    Used to determine safe mass scanning floor.
        min_events_saturation: The event count where model performance plateaus (default 40000).
                               Lower this if using a very small dataset to avoid favoring only low-mass regions.
    """
    
    if not (input_background or input_signal or input_unlabeled):
        raise ValueError("Error: At least one input file is required.")

    command = [
        f"{sys.executable}",
        "framework/tools/region_proposer.py",
        "--sigma_width", str(sigma_width),
        "--resolution_ratio", str(resolution_ratio),
        "--min_jet_pt", str(min_jet_pt),
        "--min_events_saturation", str(min_events_saturation) # <--- PASS IT
    ]

    if input_background:
        command += ["--input_background", input_background]
    if input_signal:
        command += ["--input_signal", input_signal]
    if input_unlabeled:
        command += ["--input_unlabeled", input_unlabeled]

    print(f"Worker: Executing command: {' '.join(command)}")
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )
    
    return result.stdout
    
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
    scan_start_mass: float | None = None,
    min_mass_signal_region: float | None = None,
    max_mass_signal_region: float | None = None,
    scan_end_mass: float | None = None,
    tho_21_threshold: float | None = None,
):
    """
    Tool to run LaCATHODE data preparation.

    Inputs are result of FastJet tool, fastjet tool result will give the file names.

    If fastjet tool result is background_events.jsonl and signal_events.jsonl files, use 'training' mode 
        and provide both input_background and input_signal.

    If fastjet tool result is only unlabeled_events.jsonl file, use 'inference' mode and provide input_unlabeled.
    
    As a clarification, R&D data usually has both background and signal files, so use 'training' mode.
    Real life or blackbox data usually has only unlabeled file, so use 'inference' mode.

    Always make sure scan_start_mass < min_mass_signal_region < max_mass_signal_region < scan_end_mass.
    Make sure Signal Region request is not too wide. 

    Expense is 5 GPU minutes for large datasets.
    Args:
        run_id: Unique identifier for the run. Determines the output directory. (required)
        run_mode: 'training' or 'inference'. Default is 'training'.
        input_background: Path to background data file (required for training mode).
        input_signal: Path to signal data file (required for training mode).
        input_unlabeled: Path to unlabeled data file (required for inference mode).
        shuffle_seed: Random seed for shuffling. (optional)
        training_fraction: Fraction of data to use for training. (optional)
        validation_fraction: Fraction of data to use for validation.
        scan_start_mass: Minimum mass for Sideband (SB) region in TeV. (optional)
        min_mass_signal_region: Minimum mass for Signal Region (SR) window in TeV.
        max_mass_signal_region: Maximum mass for Signal Region (SR) window in TeV.
        scan_end_mass: Maximum mass for Sideband (SB) region in TeV. (optional)
        tho_21_threshold: Threshold for Tau2/1 ratio filtering (testing feature, might give wrong results). (optional)
    """
    
    # min_mass and max_mass defaults were hidden on purpose to force LLM change them run to run properly.

    # Validation logic for required inputs based on mode
    if run_mode == 'training':
        if not input_background or not input_signal:
            raise ValueError("Error: Training mode requires both input_background and input_signal arguments.")
    elif run_mode == 'inference':
        if not input_unlabeled:
            raise ValueError("Error: Inference mode requires input_unlabeled argument.")
    else:
        raise ValueError(f"Error: Invalid run_mode '{run_mode}'. Must be 'training' or 'inference'.")

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
    if scan_start_mass is not None:
        command += ["--side_band_min", str(scan_start_mass)]
    if min_mass_signal_region is not None:
        command += ["--min_mass", str(min_mass_signal_region)]
    if max_mass_signal_region is not None:
        command += ["--max_mass", str(max_mass_signal_region)]
    if scan_end_mass is not None:
        command += ["--side_band_max", str(scan_end_mass)]
    if tho_21_threshold is not None:
        command += ["--tho_21_threshold", str(tho_21_threshold)]


    print(f"Worker: Executing command: {' '.join(command)}")
    # Run the subprocess
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )
    
    # Fallback if no tool_result tag, return full stdout
    return result.stdout
    

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

    Expense is 20-40 GPU minutes for large datasets.
    
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

    print(f"Worker: Executing command: {' '.join(command)}")
    # Run the subprocess
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )
    
    # Fallback if no tool_result tag, return full stdout
    return result.stdout
    
    
def lacathode_oracle_tool(
    inference_file: str,
    run_id: str,
    data_dir: str | None = None,
    model_dir: str | None = None,
):
    """
    Tool to run the LaCATHODE Oracle Inference.

    This tool applies a trained LaCATHODE model to new data to generate anomaly scores.
    It requires access to the original training data directory to calibrate its scalers.

    model_dir is result of LaCATHODE training tool.
    inference_file is result of LaCATHODE preparation tool.

    Expense is 5 GPU minutes for large datasets.
    
    Args:
        inference_file: Path to the data file to predict on. 
                        e.g. 'toolout/lacathode_prepared_data/{run_id}/innerdata_combined.npy'
        run_id: The ID of the run (e.g., 'run_001'). Used to locate models and save outputs.
        data_dir: (Optional) Override path to original training data. 
                  Defaults to 'toolout/lacathode_prepared_data/{run_id}/'.
        model_dir: (Optional) Override path to trained models.
                   Defaults to 'toolout/lacathode_trained_models/{run_id}/'.
    """
    
    if not inference_file:
        raise ValueError("Error: inference_file is a required argument.")

    # 1. Infer Paths from run_id if not provided
    if data_dir is None:
        data_dir = f"toolout/lacathode_prepared_data/{run_id}/"
    if model_dir is None:
        model_dir = f"toolout/lacathode_trained_models/{run_id}/"

    command = [
        f"{sys.executable}",
        "framework/tools/lacathode_oracle.py",
        "--inference_file", inference_file
    ]

    # 2. Define Explicit Output Path
    # We save directly to the model directory, but as a full path
    output_file = os.path.join(model_dir, "inference_scores.npy")

    if data_dir:
        command += ["--data_dir", data_dir]
    if model_dir:
        command += ["--model_dir", model_dir]
    
    # Pass the FULL PATH now that the script accepts it directly
    command += ["--output_file", output_file]

    print(f"Worker: Executing command: {' '.join(command)}")
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )
    
    return result.stdout

def lacathode_report_generator_tool(
    data_file: str | None = None,
    scores_file: str | None = None,
    output_file: str | None = None,
    top_percentile: float | None = None,
    bin_count: int | None = None,
    min_events_per_bin: int | None = None,
):
    """
    Tool to generate a human-readable anomaly report using lacathode_report_generator.py.

    Data file is result of LaCATHODE preparation tool.
    Scores file is result of LaCATHODE Oracle tool.
    
    This tool analyzes the anomaly scores generated by the Oracle, identifies 
    excess regions (hotspots) in the mass spectrum, and lists the top candidate events.
    
    Expense is minimal, typically a few CPU minutes.
    
    Args:
        data_file: Path to the input data file (numpy .npy format). 
        scores_file: Path to the anomaly scores file (numpy .npy format).
        output_file: Path to save the generated report (default "llm_enhanced_report.txt").
        top_percentile: Percentile threshold to define anomaly candidates. Default 99.0 But you can experiment.
        bin_count: Number of bins for mass histogram. Default 18 But you can experiment.
        min_events_per_bin: Minimum events required in a bin to calculate excess factor (filters noise). Default 200.
    
    Returns:
        The content of the generated report.
    """
    
    if not data_file or not scores_file:
        raise ValueError("Error: Both data_file and scores_file are required arguments.")

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
    if min_events_per_bin is not None:
        command += ["--min_events_per_bin", str(min_events_per_bin)]


    print(f"Worker: Executing command: {' '.join(command)}")
    # Run the subprocess
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )
    
    return result.stdout
    
def query_knowledge_base_tool(
    search_query: str,
):
    """
    Tool to query the internal knowledge base (PDFs/Articles) using RAG Engine.

    Args:
        search_query: The topic or question to search for.
                      Example: "What is the mass of the Top Quark?"
    Returns:
        The retrieved information from the knowledge base.
    """
    command = [
        f"{sys.executable}",
        "framework/rag_engine.py",
        "--ingest",
        "--query", search_query,
    ]

    print(f"Worker: Executing command: {' '.join(command)}")

    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True, 
        env=os.environ
    )

    return result.stdout

def query_gemma_cloud_tool(
    query: str,
):
    """
    Tool to query Google Gemma Cloud.

    Use only after trying query_knowledge_base_tool first.

    Paid service, use sparingly.
    
    Args:
        query: The input query string to send to Gemma.
    """

    return query_gemma_cloud(query)

def python_repl_tool(
    code: str,
    run_as_subprocess: bool = False,
):
    """
    Tool to execute Python code with restricted WRITE permissions.
    - READ: Allowed everywhere (load data, configs, etc.)
    - WRITE: Allowed ONLY in './toolout/repl/'.
    - BLOCKED: os.system, shell commands.

    Available libraries (already imported in the REPL environment):
        numpy as np, pandas as pd, matplotlib.pyplot as plt, os, torch, scipy

    Args:
        code: The Python code to execute.
        run_as_subprocess: If True, runs the code in a separate subprocess for added isolation and performance.
                           If False, runs directly in the current process with safety wrappers.
    """
    from framework.tools.python_repl import python_repl_tool as repl_tool

    if run_as_subprocess:
        command = [
            f"{sys.executable}",
            "framework/tools/python_repl_subprocess.py",
            "--code", code,
        ]

        print(f"Worker: Executing command: {' '.join(command)}")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=os.environ
        )
        
        return result.stdout
    else:
        print(f"Worker: Executing Safe Python Code directly...")
        return repl_tool(code)