import os
import numpy as np
import glob
import json
from argparse import ArgumentParser
from collections import defaultdict

"""
LaCATHODE Aggregator
====================
Merges outputs from sliding window tests into a single global result with
proper uncertainty quantification for scientific analysis.

THE LOGIC:
Since sliding windows overlap (e.g. 2000-2400 and 2100-2500), simply concatenating
the files would double-count events in the overlap region (2100-2400).
We perform "Kinematic De-duplication" with uncertainty propagation:

1. Event Identification: Use a composite key of (mass, feature_hash) for robustness.
   Mass alone can have collisions; adding a hash of key features ensures uniqueness.
   
2. Score Aggregation: For each unique event seen in N windows:
   - Compute mean score (consensus prediction)
   - Compute standard deviation (model uncertainty across windows)
   - Track observation count N (coverage metadata)
   
3. Weighted Scoring (Optional): Events in overlapping regions get more predictions,
   which naturally reduces their variance. This is statistically correct behavior.

4. Output: Produces aggregated data + scores with uncertainty metadata for
   proper error propagation in downstream statistical analysis.
"""

parser = ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True,
                    help="Root folder containing the sliding window run subdirectories.")
parser.add_argument("--output_dir", type=str, default="toolout/aggregated_results/",
                    help="Destination for the merged .npy files.")
parser.add_argument("--filename_pattern", type=str, default="win_*",
                    help="Glob pattern to identify run folders (e.g. 'win_*').")
parser.add_argument("--mass_precision", type=int, default=6,
                    help="Decimal precision for mass-based event matching (default: 6 = ~1 eV resolution).")
parser.add_argument("--use_feature_hash", action="store_true", default=True,
                    help="Use feature hash in addition to mass for event identification (recommended).")

args = parser.parse_args()

def compute_event_key(mass, features, mass_precision=6, use_feature_hash=True):
    """
    Generate a unique identifier for an event.
    
    Using mass alone risks collisions in large datasets. We optionally include
    a hash of discriminating features (dR, mass_diff, tau ratios) for robustness.
    
    Args:
        mass: Invariant mass (TeV)
        features: Full feature vector for the event
        mass_precision: Decimal places to round mass (6 = ~1 eV precision)
        use_feature_hash: Whether to include feature hash in key
        
    Returns:
        Tuple key that uniquely identifies this event
    """
    mass_key = round(float(mass), mass_precision)
    
    if use_feature_hash and features is not None and len(features) > 3:
        # Use stable features that shouldn't vary between windows:
        # dR (col 2), mass_diff (col 3), tau ratios (cols 14, 25 if available)
        stable_indices = [2, 3]  # dR, mass_diff
        if len(features) > 14:
            stable_indices.append(14)  # j1_tau2_over_tau1
        if len(features) > 25:
            stable_indices.append(25)  # j2_tau2_over_tau1
            
        # Round to avoid floating point noise, then hash
        stable_vals = tuple(round(float(features[i]), 4) for i in stable_indices if i < len(features))
        return (mass_key, stable_vals)
    
    return (mass_key,)


def aggregate_results(source_dir, output_dir, pattern, mass_precision=6, use_feature_hash=True):
    print(f"--- Aggregating Sliding Window Results from {source_dir} ---")
    print(f"    Mass precision: {mass_precision} decimals")
    print(f"    Feature hashing: {'enabled' if use_feature_hash else 'disabled'}")
    
    run_dirs = glob.glob(os.path.join(source_dir, pattern))
    if not run_dirs:
        print(f"No directories found matching '{pattern}'. Check your path.")
        return False

    # Registry: { event_key: {'scores': [...], 'features': array, 'windows': [...]} }
    # Stores all predictions plus metadata for uncertainty quantification
    event_registry = defaultdict(lambda: {'scores': [], 'features': None, 'windows': []})
    
    files_processed = 0
    total_events_read = 0
    window_metadata = []  # Track which windows were processed

    print(f"Found {len(run_dirs)} potential run directories. Processing...")

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        
        # --- LOCATE FILES (with recursive search) ---
        # Score file is directly in run_dir
        score_file = os.path.join(run_dir, "scores.npy")
        
        # Data file might be in a subdirectory (run_dir/job_id/)
        # Search recursively for innerdata_inference.npy
        data_files = glob.glob(os.path.join(run_dir, "**", "*innerdata_inference.npy"), recursive=True)
        if not data_files:
            data_files = glob.glob(os.path.join(run_dir, "*innerdata_inference.npy"))
        if not data_files:
            # Also try direct path without glob
            potential_path = os.path.join(run_dir, run_name, "innerdata_inference.npy")
            if os.path.exists(potential_path):
                data_files = [potential_path]
        
        # Skip failed/incomplete runs
        if not os.path.exists(score_file):
            print(f"  [SKIP] {run_name}: No scores.npy found")
            continue
        if not data_files:
            print(f"  [SKIP] {run_name}: No innerdata_inference.npy found")
            continue

        try:
            scores = np.load(score_file)
            data = np.load(data_files[0])
            
            # Ensure scores is 1D
            scores = scores.flatten()
            
            # Sanity check: Row counts must match exactly
            if len(scores) != len(data):
                print(f"  [WARN] Skipping {run_name}: Size mismatch (Scores={len(scores)}, Data={len(data)})")
                continue
            
            # Extract window bounds from run name if possible (e.g., "win_2000_2400_rep0")
            window_info = run_name
                
            # --- DE-DUPLICATION WITH FULL FEATURE PRESERVATION ---
            for i, (score, features) in enumerate(zip(scores, data)):
                mass = features[0]  # Column 0 is always invariant mass
                
                event_key = compute_event_key(mass, features, mass_precision, use_feature_hash)
                
                event_registry[event_key]['scores'].append(float(score))
                event_registry[event_key]['windows'].append(window_info)
                
                # Store features from first observation (they should be identical)
                if event_registry[event_key]['features'] is None:
                    event_registry[event_key]['features'] = features.copy()
                
            files_processed += 1
            total_events_read += len(scores)
            window_metadata.append({'name': run_name, 'events': len(scores)})
            print(f"  [OK] {run_name}: {len(scores)} events")
            
        except Exception as e:
            print(f"  [ERR] Failed to process {run_dir}: {e}")

    # --- VALIDATION ---
    unique_events = len(event_registry)
    if unique_events == 0:
        print("\nAggregation Failed: No valid data found.")
        return False

    # --- STATISTICS ---
    observation_counts = [len(v['scores']) for v in event_registry.values()]
    
    print(f"\n{'='*50}")
    print(f"AGGREGATION STATISTICS")
    print(f"{'='*50}")
    print(f"  Windows Processed:    {files_processed}")
    print(f"  Raw Events Read:      {total_events_read:,}")
    print(f"  Unique Events:        {unique_events:,}")
    print(f"  De-duplication Rate:  {(1 - unique_events/total_events_read)*100:.1f}%")
    print(f"  Overlap Factor:       {total_events_read / unique_events:.2f}x (avg observations per event)")
    print(f"  Observations/Event:")
    print(f"    - Min: {min(observation_counts)}")
    print(f"    - Max: {max(observation_counts)}")
    print(f"    - Mean: {np.mean(observation_counts):.2f}")
    print(f"{'='*50}")

    # --- RECONSTRUCTION WITH UNCERTAINTY ---
    print("\nReconstructing global spectrum with uncertainty quantification...")
    
    # Sort by mass for cleaner output
    sorted_keys = sorted(event_registry.keys(), key=lambda k: k[0])
    
    n_events = len(sorted_keys)
    n_features = len(next(iter(event_registry.values()))['features'])
    
    # Output arrays
    final_data = np.zeros((n_events, n_features), dtype=np.float64)
    final_scores = np.zeros(n_events, dtype=np.float64)
    final_uncertainties = np.zeros(n_events, dtype=np.float64)
    final_obs_counts = np.zeros(n_events, dtype=np.int32)
    
    for i, key in enumerate(sorted_keys):
        entry = event_registry[key]
        score_list = np.array(entry['scores'])
        
        # Store original features
        final_data[i] = entry['features']
        
        # Consensus score: mean across all window observations
        final_scores[i] = np.mean(score_list)
        
        # Uncertainty: standard deviation (0 if only one observation)
        if len(score_list) > 1:
            final_uncertainties[i] = np.std(score_list, ddof=1)  # ddof=1 for sample std
        else:
            final_uncertainties[i] = np.nan  # Mark as unknown uncertainty
            
        final_obs_counts[i] = len(score_list)

    # --- SAVE OUTPUTS ---
    os.makedirs(output_dir, exist_ok=True)
    
    out_data_path = os.path.join(output_dir, "aggregated_data.npy")
    out_score_path = os.path.join(output_dir, "aggregated_scores.npy")
    out_uncertainty_path = os.path.join(output_dir, "aggregated_uncertainties.npy")
    out_counts_path = os.path.join(output_dir, "aggregated_observation_counts.npy")
    out_metadata_path = os.path.join(output_dir, "aggregation_metadata.json")
    
    # Save with float64 for scientific precision
    np.save(out_data_path, final_data.astype(np.float64))
    np.save(out_score_path, final_scores.astype(np.float64))
    np.save(out_uncertainty_path, final_uncertainties.astype(np.float64))
    np.save(out_counts_path, final_obs_counts)
    
    # Save metadata for reproducibility
    metadata = {
        'source_dir': source_dir,
        'pattern': pattern,
        'mass_precision': mass_precision,
        'use_feature_hash': use_feature_hash,
        'windows_processed': files_processed,
        'total_events_read': total_events_read,
        'unique_events': unique_events,
        'overlap_factor': total_events_read / unique_events,
        'observation_stats': {
            'min': int(min(observation_counts)),
            'max': int(max(observation_counts)),
            'mean': float(np.mean(observation_counts)),
            'std': float(np.std(observation_counts))
        },
        'windows': window_metadata
    }
    
    with open(out_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSuccess! Output files saved to {output_dir}:")
    print(f"  - aggregated_data.npy          : Full feature matrix ({n_events} x {n_features})")
    print(f"  - aggregated_scores.npy        : Consensus anomaly scores")
    print(f"  - aggregated_uncertainties.npy : Score std. dev. across windows")
    print(f"  - aggregated_observation_counts.npy : # of windows per event")
    print(f"  - aggregation_metadata.json    : Processing metadata")
    
    return True


if __name__ == "__main__":
    success = aggregate_results(
        args.source_dir, 
        args.output_dir, 
        args.filename_pattern,
        args.mass_precision,
        args.use_feature_hash
    )
    exit(0 if success else 1)