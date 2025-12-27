import os
import json
import time
import subprocess
import sys

from framework.tools.worker_tools import fastjet_tool

TOOL_REGISTRY = {
    "fastjet_tool": fastjet_tool,
    # "another_tool_name": another_tool_function
}

PENDING_DIR = "jobs/pending"
COMPLETED_DIR = "jobs/completed"
os.makedirs(COMPLETED_DIR, exist_ok=True)

def execute_tool(name, args):
    print(f"Worker: Received task '{name}' with args: {args}")
    
    if name not in TOOL_REGISTRY:
        return f"Error: Tool '{name}' is not registered in the worker."
    
    # Get the function
    tool_func = TOOL_REGISTRY[name]
    
    # Execute with unpacked arguments
    try:
        print(f"Worker: Executing tool function '{name}'...")
        print(f"Worker: Tool function args: {args}")
        result = tool_func(**args)
        return result
    except TypeError as e:
        return f"Argument Error: {str(e)}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

def wake_up_bot(job_id):
    """
    Restarts the main application with the context of the completed job.
    """
    print(f"Waking up bot for job {job_id}...")
    
    # 1. Inherit Environment Variables
    # Critical for CUDA_VISIBLE_DEVICES, API Keys, Python path, etc.
    env = os.environ.copy()
    
    # 2. Launch the process
    # We use Popen so this script continues (or can exit), 
    # letting the bot run independently.
    subprocess.Popen(
        [sys.executable, "-m", "framework", "--resume", job_id],
        env=env,
        cwd=os.getcwd() # Ensure we launch from the correct root
    )

def run_worker():
    print("Worker started. Monitoring jobs/pending/ ...")
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
                
                print(f"Processing Job: {data['job_id']}")

                # Run the Tool
                result = execute_tool(data["tool_name"], data["tool_args"])
                
                # Create Result Packet
                result_data = {
                    "original_state": data,
                    "tool_result": result,
                    "status": "success" # You can refine this based on result content
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
                    print(f"Queue not empty ({len(remaining_jobs)} remaining). "
                          "Holding wake-up signal to prevent process conflicts.")
                else:
                    # Only wake up if this was the last job in the queue
                    wake_up_bot(data["job_id"])
                
            time.sleep(2)
        except KeyboardInterrupt:
            print("Worker stopping...")
            break
        except Exception as e:
            print(f"Worker Loop Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    run_worker()