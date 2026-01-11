import argparse
import numpy as np
import json
import os
import sys

from framework.logger import get_logger

logger = get_logger(__name__)

def load_masses_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    masses_list = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'm_jj' in obj:
                        masses_list.append(obj['m_jj'])
                except json.JSONDecodeError:
                    continue
        return np.array(masses_list)
    elif file_path.endswith('.npy'):
        data = np.load(file_path)
        return data[:, 0] if data.ndim > 1 else data
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

def propose_regions(
    input_background=None, 
    input_signal=None, 
    input_unlabeled=None, 
    global_start=2000.0,
    global_stop=6000.0,
    window_width=400.0,
    step_size=200.0,
    trigger_threshold_pt=1200.0
):
    """
    Implements a Sliding Window Scan for CATHODE/LaCATHODE anomaly detection.
    
    Architecture Decisions:
    1. Global Sidebands: Unlike localized sidebands (e.g., +/- 200 GeV), we use the full 
       available spectrum (2-6 TeV) for background estimation. This prevents "data starvation" 
       where the Normalizing Flow fails to learn the global background probability density p(x|m).
    
    2. Bottleneck Scoring: Regions are ranked by their weakest statistical link. 
       Analysis stability is strictly limited by the region (Left SB, SR, or Right SB) 
       with the fewest events. This naturally filters out the trigger turn-on (empty Left SB) 
       and the kinematic tail (empty Right SB/SR).
    """
    
    # 1. Load Data
    masses = None
    try:
        if input_unlabeled:
            masses = load_masses_from_file(input_unlabeled)
        elif input_background and input_signal:
            bg_masses = load_masses_from_file(input_background)
            sig_masses = load_masses_from_file(input_signal)
            if len(bg_masses) > 0 and len(sig_masses) > 0:
                masses = np.concatenate([bg_masses, sig_masses])
            elif len(bg_masses) > 0:
                masses = bg_masses
            else:
                masses = sig_masses
        else:
            return {"error": "No input files provided."}

        if masses is None or len(masses) == 0:
            return {"error": "No events found."}

        # Standardization: Ensure GeV units for consistency with trigger thresholds
        if masses.max() < 100:
            masses *= 1000.0

        # Scanning Routine
        candidates = []

        # The invariant mass turn-on is roughly 2x the jet pT trigger threshold.
        safe_mass_floor = 2.0 * trigger_threshold_pt
        
        # Enforce Minimum Anchor Width (0.5 TeV = 500 GeV)
        # The SR cannot start until we have at least 500 GeV of Sideband to its left.
        min_anchor_gev = 500.0
        
        # Start Constraint: Must be > safe floor AND > global_start + anchor
        min_valid_start = max(safe_mass_floor, global_start + min_anchor_gev)
        
        if min_valid_start > global_start:
            logger.info(f"Adjusting scan start to {min_valid_start} GeV to ensure {min_anchor_gev} GeV Left Anchor.")
            current_sr_start = min_valid_start
        else:
            current_sr_start = global_start

        # Stop Constraint: Must leave room for Right Anchor
        # The scan must stop early enough so that (global_stop - sr_end) >= 500
        effective_global_stop = global_stop - min_anchor_gev

        while current_sr_start + window_width <= effective_global_stop:
            sr_end = current_sr_start + window_width
            sr_center = (current_sr_start + sr_end) / 2
            
            # Sideband Definition:
            # We fix sidebands to the global analysis limits. 
            # The SR creates a "hole" in this global dataset.
            sb_min = global_start
            sb_max = global_stop
            
            # Event Counting
            left_mask = (masses >= sb_min) & (masses < current_sr_start)
            sr_mask = (masses >= current_sr_start) & (masses < sr_end)
            right_mask = (masses >= sr_end) & (masses < sb_max)
            
            left_count = np.sum(left_mask)
            sr_count = np.sum(sr_mask)
            right_count = np.sum(right_mask)

            # Scoring Logic
            # Metric: Minimum Event Count (Bottleneck)
            # 
            # Why: 
            # - Trigger Turn-on (Low Mass): Left SB count drops to zero. Model cannot interpolate from left.
            # - Kinematic Tail (High Mass): SR/Right SB count drops. Statistical significance vanishes.
            # 
            # This metric forces the selection of the "sweet spot" (typically 3-4 TeV) where
            # the trigger efficiency plateau meets the falling cross-section.
            bottleneck_stat = min(left_count, right_count, sr_count)
            
            # Filter unstable regions (arbitrary floor to prevent singular covariance matrices in training)
            if bottleneck_stat > 100: 
                rec = {
                    "id": f"SR_{int(sr_center)}",
                    "focus_mass_gev": int(sr_center),
                    "quality_score": int(bottleneck_stat),
                    "tool_parameters": {
                        "scan_start_mass": round(sb_min / 1000.0, 4),      
                        "min_mass_signal_region": round(current_sr_start / 1000.0, 4), 
                        "max_mass_signal_region": round(sr_end / 1000.0, 4),   
                        "scan_end_mass": round(sb_max / 1000.0, 4)         
                    },
                    "stats": {
                        "sr_events": int(sr_count),
                        "left_sb_events": int(left_count),
                        "right_sb_events": int(right_count),
                        "bottleneck": int(bottleneck_stat)
                    }
                }
                candidates.append(rec)

            current_sr_start += step_size

        # Output
        # Return sorted by robustness (Descending Bottleneck Score)
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return candidates[:15]

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_background', type=str)
    parser.add_argument('--input_signal', type=str)
    parser.add_argument('--input_unlabeled', type=str)
    
    # Scan Configuration
    parser.add_argument("--global_start", type=float, default=2000.0)
    parser.add_argument("--global_stop", type=float, default=6000.0)
    parser.add_argument("--window_width", type=float, default=400.0)
    parser.add_argument("--step_size", type=float, default=200.0)
    parser.add_argument("--trigger_threshold_pt", type=float, default=1200.0)

    args = parser.parse_args()

    results = propose_regions(
        input_background=args.input_background,
        input_signal=args.input_signal,
        input_unlabeled=args.input_unlabeled,
        global_start=args.global_start,
        global_stop=args.global_stop,
        window_width=args.window_width,
        step_size=args.step_size,
        trigger_threshold_pt=args.trigger_threshold_pt
    )

    print("<tool_result>")
    print(json.dumps(results, indent=2))
    print("</tool_result>")