import os
import numpy as np
import glob
from argparse import ArgumentParser

"""
LaCATHODE Aggregator
====================
Merges outputs from sliding window tests into a single global result.

THE LOGIC:
Since sliding windows overlap (e.g. 2000-2400 and 2100-2500), simply concatenating
the files would double-count events in the overlap region (2100-2400).
Instead, we perform "Kinematic De-duplication":
1. We assume Invariant Mass (Column 0) acts as a unique ID for each event.
   (In high-precision physics data, collisions is negligible).
2. We map every event to its Mass -> [List of Scores from all windows].
3. We compute the mean score for each unique event.

This produces a single, statistically robust spectrum covering the full range.
"""

parser = ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True,
                    help="Root folder containing the sliding window run subdirectories.")
parser.add_argument("--output_dir", type=str, default="toolout/aggregated_results/",
                    help="Destination for the merged .npy files.")
parser.add_argument("--filename_pattern", type=str, default="win_*",
                    help="Glob pattern to identify run folders (e.g. 'win_*').")

args = parser.parse_args()

def aggregate_results(source_dir, output_dir, pattern):
    print(f"--- Aggregating Sliding Window Results from {source_dir} ---")
    
    run_dirs = glob.glob(os.path.join(source_dir, pattern))
    if not run_dirs:
        print(f"No directories found matching '{pattern}'. Check your path.")
        return

    # Registry: { mass_key (float): [score_1, score_2, ...] }
    # Used to collect multiple anomaly scores for the same event across different windows.
    event_registry = {}
    
    files_processed = 0
    total_events_read = 0

    print(f"Found {len(run_dirs)} potential run directories. Processing...")

    for run_dir in run_dirs:
        # Locate the Score file (Output of Oracle)
        score_file = os.path.join(run_dir, "scores.npy")
        
        # Locate the Data file (Input to Oracle, contains Mass)
        # We need this because scores.npy is just a 1D array; we need Mass to identify events.
        # Naming might vary (e.g., 'win_2000_rep0_innerdata...'), so we glob.
        data_files = glob.glob(os.path.join(run_dir, "*innerdata_inference.npy"))
        if not data_files:
            # Try generic fallback if specific ID wasn't used
            data_files = glob.glob(os.path.join(run_dir, "innerdata_inference.npy"))
        
        # skip failed/incomplete runs
        if not os.path.exists(score_file) or not data_files:
            continue

        try:
            scores = np.load(score_file)
            data = np.load(data_files[0]) # Use first match
            
            # Sanity check: Row counts must match exactly
            if len(scores) != len(data):
                print(f"[WARN] Skipping {os.path.basename(run_dir)}: Size mismatch (Scores={len(scores)}, Data={len(data)})")
                continue
                
            # Column 0 is always Invariant Mass
            masses = data[:, 0]
            
            # --- DE-DUPLICATION CORE ---
            # Zip Mass + Score together.
            # If an event (Mass X) was seen in a previous window, we add this new score to its list.
            for m, s in zip(masses, scores):
                # Round to 5 decimals to handle floating point drift between files.
                # (e.g. 3000.0000001 vs 3000.0000002)
                m_key = round(float(m), 5) 
                
                if m_key not in event_registry:
                    event_registry[m_key] = []
                event_registry[m_key].append(s)
                
            files_processed += 1
            total_events_read += len(scores)
            
        except Exception as e:
            print(f"[ERR] Failed to process {run_dir}: {e}")

    # --- SUMMARY ---
    unique_events = len(event_registry)
    if unique_events == 0:
        print("Aggregation Failed: No valid data found.")
        return

    print(f"\nAggregation Stats:")
    print(f"  > Files Processed: {files_processed}")
    print(f"  > Raw Events Read: {total_events_read}")
    print(f"  > Unique Events:   {unique_events} (after de-duplication)")
    print(f"  > Overlap Factor:  {total_events_read / unique_events:.2f}x (avg windows per event)")

    # --- RECONSTRUCTION ---
    print("Reconstructing global spectrum...")
    
    final_masses = []
    final_scores = []
    
    # Sort keys so the final file is ordered by Mass (cleaner for plotting)
    sorted_masses = sorted(event_registry.keys())
    
    for m in sorted_masses:
        score_list = event_registry[m]
        
        # The Consensus Score: Average of all predictions for this event.
        # This reduces variance significantly compared to a single run.
        avg_score = np.mean(score_list)
        
        final_masses.append(m)
        final_scores.append(avg_score)
        
    # Convert to standard numpy format
    final_masses = np.array(final_masses, dtype=np.float32)
    final_scores = np.array(final_scores, dtype=np.float32)
    
    # Create the "Mock" Data array required by Report Generator
    # It expects shape (N, 27) where Col 0 is Mass.
    # We zero-fill the rest (features) since they aren't needed for the final Bump Hunt plot.
    n_cols = 27 
    final_data = np.zeros((len(final_masses), n_cols), dtype=np.float32)
    final_data[:, 0] = final_masses

    # --- SAVE OUTPUTS ---
    os.makedirs(output_dir, exist_ok=True)
    
    out_data_path = os.path.join(output_dir, "aggregated_data.npy")
    out_score_path = os.path.join(output_dir, "aggregated_scores.npy")
    
    np.save(out_data_path, final_data)
    np.save(out_score_path, final_scores)
    
    print(f"\nSuccess! Merged files saved to:")
    print(f"  Data:   {out_data_path}")
    print(f"  Scores: {out_score_path}")

if __name__ == "__main__":
    aggregate_results(args.source_dir, args.output_dir, args.filename_pattern)