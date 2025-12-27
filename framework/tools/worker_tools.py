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
    If you think dataset is NOT R&D. Meaning it is something like blackbox or experiment result set no_label_input=True.
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
    