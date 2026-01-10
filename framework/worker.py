import os
import json
import time
import subprocess
import sys
import argparse

from framework.tools.worker_tools import fastjet_tool, lacathode_preparation_tool, \
    lacathode_training_tool, lacathode_oracle_tool, lacathode_report_generator_tool, \
    query_knowledge_base_tool, query_gemma_cloud_tool, propose_signal_regions_tool, \
    python_repl_tool, isolation_forest_tool
from framework.logger import get_logger

logger = get_logger("Worker")

TOOL_REGISTRY = {
    "fastjet_tool": fastjet_tool,
    "lacathode_preparation_tool": lacathode_preparation_tool,
    "lacathode_training_tool": lacathode_training_tool,
    "lacathode_oracle_tool": lacathode_oracle_tool,
    "lacathode_report_generator_tool": lacathode_report_generator_tool,
    "query_knowledge_base_tool": query_knowledge_base_tool,
    "query_gemma_cloud_tool": query_gemma_cloud_tool,
    "propose_signal_regions_tool": propose_signal_regions_tool,
    "python_repl_tool": python_repl_tool,
    "isolation_forest_tool": isolation_forest_tool,
}

PENDING_DIR = os.path.join("jobs", "pending")
COMPLETED_DIR = os.path.join("jobs", "completed")
LOG_DIR = os.path.join("jobs", "logs")
os.makedirs(COMPLETED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)

# Track current bot process for wake-up management
CURRENT_BOT_PROCESS = None

def shutdown_bot():
    """
    Checks if a bot instance is currently running and shuts it down.
    DISABLED for now due switching to external ollama server.
    """
    global CURRENT_BOT_PROCESS
    
    if CURRENT_BOT_PROCESS is not None:
        # Check if process is still alive (poll returns None if alive)
        if CURRENT_BOT_PROCESS.poll() is None:
            logger.info(f"Worker: Shutting down active bot instance (PID: {CURRENT_BOT_PROCESS.pid})...")
            try:
                # Try graceful termination first
                CURRENT_BOT_PROCESS.terminate()
                CURRENT_BOT_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.info("Worker: Process unresponsive, forcing kill...")
                CURRENT_BOT_PROCESS.kill()
                CURRENT_BOT_PROCESS.wait()
        
        # Clear the reference
        CURRENT_BOT_PROCESS = None

def execute_tool(name, args):
    logger.info(f"Worker: Received task '{name}' with args: {args}")
    
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Error: Tool '{name}' is not registered in the worker.")
    
    # Get the function
    tool_func = TOOL_REGISTRY[name]

    # Kill any active bot before running tool
    #shutdown_bot()
    
    # Execute with unpacked arguments
    logger.info(f"Worker: Executing tool function '{name}'...")
    logger.info(f"Worker: Tool function args: {args}")
    result = tool_func(**args)
    return result

def start_up_bot(job_id: str | None):
    """
    Starts the main bot process.
    If a job_id is provided, the bot will resume that job.
    """
    logger.info(f"Worker: Starting up main bot process (resume job: {job_id})...")
    global CURRENT_BOT_PROCESS

    # Kill any existing bot process first
    #shutdown_bot()
    
    # Inherit Environment Variables
    # Critical for CUDA_VISIBLE_DEVICES, API Keys, Python path, etc.
    env = os.environ.copy()
    
    # Launch the process
    # We use Popen so this script continues (or can exit), 
    # letting the bot run independently.
    launch_command = [sys.executable, "-m", "framework", "--framework-only"]
    if job_id:
        launch_command += ["--resume", job_id]

    CURRENT_BOT_PROCESS = subprocess.Popen(
        launch_command,
        env=env,
        cwd=os.getcwd() # Ensure we launch from the correct root
    )

def find_cached_result(tool_name, tool_args):
    """
    Scans completed jobs to find an exact match for tool_name and tool_args.
    Returns the previous result if found, otherwise None.
    """
    if not os.path.exists(COMPLETED_DIR):
        return None

    logger.info(f"Debug Cache: Scanning for previous runs of {tool_name}...")
    
    # Iterate over all completed job files
    for fname in os.listdir(COMPLETED_DIR):
        if not fname.endswith(".json"):
            continue
            
        fpath = os.path.join(COMPLETED_DIR, fname)
        try:
            with open(fpath, "r") as f:
                cached_data = json.load(f)
                
            # Check if valid structure
            if "original_state" not in cached_data:
                continue

            cached_state = cached_data["original_state"]
            
            # Check for exact match (Name + Args)
            # comparing dicts directly works in Python (order doesn't matter)
            # check status to ensure it was a successful run
            if (cached_state.get("tool_name") == tool_name and 
                cached_state.get("tool_args") == tool_args and
                cached_data.get("status") == "success"):
                
                logger.info(f"Debug Cache: HIT found in {fname}! Skipping execution.")
                return cached_data.get("tool_result")
                
        except (json.JSONDecodeError, OSError):
            continue
            
    return None

def run_worker(cache_tools=[]):
    """
    Main worker loop that monitors the pending jobs directory,
    executes tools, and saves results.

    Caching can be enabled for specific tools by providing their names
    in the cache_tools list.

    Caching checks for previous identical runs (same tool name and args)
    in the completed jobs directory and reuses results if found.
    
    :param cache_tools: List of tool names for which caching is enabled. Use ['*'] or ['all'] to enable for all tools.
    """
    logger.info("Worker started. Monitoring jobs/pending/ ...")
    while True:
        try:
            jobs = [f for f in os.listdir(PENDING_DIR) if f.endswith(".json")]
            
            for job_file in jobs:
                job_path = os.path.join(PENDING_DIR, job_file)
                
                # Check if file is fully written (simple lock check)
                try:
                    with open(job_path, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    continue # File might be writing still
                
                current_job_id = data['job_id']
                tool_name = data["tool_name"]
                tool_args = data["tool_args"]
                
                logger.info(f"Processing Job: {current_job_id}")

                result = None
                
                # --- DEBUG CACHE MODE START ---
                if tool_name in cache_tools or '*' in cache_tools or 'all' in cache_tools:
                    cached_result = find_cached_result(tool_name, tool_args)
                    if cached_result is not None:
                        result = cached_result
                        logger.info("Worker: Used cached result.")
                # --- DEBUG CACHE MODE END ---

                # If no cache hit (or mode disabled), run for real
                status = "success"
                if result is None:
                    try:
                        result = execute_tool(tool_name, tool_args)
                    except subprocess.CalledProcessError as e:
                        # Capture detailed subprocess error for the LLM
                        status = "error"
                        result = f"Tool Execution Failed (Exit Code: {e.returncode}).\n\nSTDERR:\n{e.stderr}"
                        if e.stdout:
                            result += f"\n\nSTDOUT:\n{e.stdout}"
                            
                    except Exception as e:
                        # Capture generic python errors (like ValueError from validation)
                        status = "error"
                        result = f"Tool Execution Error: {str(e)}"

                if result:
                    # Save stdout to log file for reference
                    log_path = os.path.join(LOG_DIR, f"{current_job_id}_run_log.txt")
                    with open(log_path, "w") as log_file:
                        log_file.write(result)
                
                # If success, ensure result is wrapped in <tool_result> tags
                if status == "success" and result and isinstance(result, str):
                    start_tag = "<tool_result>"
                    end_tag = "</tool_result>"
                    
                    if start_tag in result and end_tag in result:
                        # Extract content between tags
                        start_idx = result.find(start_tag) + len(start_tag)
                        end_idx = result.find(end_tag)
                        result = result[start_idx:end_idx].strip()
                    else:
                        result = result.strip()
                
                # Create Result Packet
                result_data = {
                    "original_state": data,
                    "tool_result": result,
                    "status": status,
                }
                
                # Save to Completed
                result_path = os.path.join(COMPLETED_DIR, job_file)
                with open(result_path, "w") as f:
                    json.dump(result_data, f, indent=2)
                
                # Cleanup Pending
                os.remove(job_path)
                
                # Conditional Wake-Up
                # Check if there are MORE jobs currently in the folder.
                # We re-list the directory to catch anything that arrived during execution.
                remaining_jobs = [f for f in os.listdir(PENDING_DIR) if f.endswith(".json")]
                
                if remaining_jobs:
                    logger.info(f"Queue not empty ({len(remaining_jobs)} remaining). ")
                
            time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Worker stopping...")
            break
        except Exception as e:
            logger.warning(f"Worker Loop Error: {e}")
            time.sleep(2)