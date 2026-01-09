import argparse
import numpy as np
import json
import os
import sys

def load_masses_from_file(file_path):
    """
    Helper function to load 'm_jj' (invariant mass) from a file.
    Supports .jsonl (FastJet output) and .npy formats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    masses_list = []

    # Case A: JSONL
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

    # Case B: Numpy
    elif file_path.endswith('.npy'):
        data = np.load(file_path)
        if data.ndim > 1:
            return data[:, 0]
        else:
            return data
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use .jsonl or .npy")

def propose_regions(
    input_background=None, 
    input_signal=None, 
    input_unlabeled=None, 
    resolution_ratio=0.03, 
    sigma_width=4.0, 
    min_jet_pt=1200.0,
    min_events_saturation=40000
):
    """
    Analyzes input data files to propose valid LaCATHODE training windows.
    Returns a RANKED list of candidates, prioritizing HIGH MASS regions.
    """
    
    masses = None
    
    try:
        # 1. Load Data
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
        elif input_background:
            masses = load_masses_from_file(input_background)
        elif input_signal:
            masses = load_masses_from_file(input_signal)
        else:
            return {"error": "No input files provided."}

        if masses is None or len(masses) == 0:
            return {"error": "No events found."}

        # 2. Unit Handling
        if masses.max() < 100:
            masses *= 1000.0
            
        max_mass_dataset = masses.max()

        # 3. Derive Safe Mass Threshold from Sensor Info (min_pt)
        # Rule of Thumb: Safe Mass Start ~ 1.1 * min_jet_pt to avoid turn-on artifacts
        safe_mass_floor = min_jet_pt * 1.1

        # 4. Scanning & Scoring
        candidates = []
        current_center = safe_mass_floor * 1.15 

        while current_center < max_mass_dataset:
            # Window Calculation
            sigma = current_center * resolution_ratio
            sr_width = sigma * sigma_width
            
            sr_start = current_center - (sr_width / 2)
            sr_end = current_center + (sr_width / 2)
            
            window_pad = sr_width * 1.5 
            sb_min = sr_start - window_pad
            sb_max = sr_end + window_pad

            # Trigger Constraint: Sideband must be in safe physics region
            if sb_min < safe_mass_floor:
                current_center += (sr_width * 0.2)
                continue
            if sb_max > max_mass_dataset:
                break

            # Counting
            left_sb_mask = (masses >= sb_min) & (masses < sr_start)
            sr_mask = (masses >= sr_start) & (masses < sr_end)
            right_sb_mask = (masses >= sr_end) & (masses < sb_max)
            
            left_count = np.sum(left_sb_mask)
            sr_count = np.sum(sr_mask)
            right_count = np.sum(right_sb_mask)

            # --- SCORING LOGIC ---
            min_sideband = min(left_count, right_count)
            robustness_score = min(min_sideband, min_events_saturation) # Saturation cap
            physics_score = robustness_score * current_center # Bias towards High Mass
            
            if left_count > 200 and right_count > 200 and sr_count > 10:
                rec = {
                    "id": f"SR_{int(current_center)}",
                    "focus_mass_gev": int(current_center),
                    "quality_score": int(physics_score),
                    "rank_comment": "", 
                    "tool_parameters": {
                        "scan_start_mass": round(sb_min / 1000.0, 4),      
                        "min_mass_signal_region": round(sr_start / 1000.0, 4), 
                        "max_mass_signal_region": round(sr_end / 1000.0, 4),   
                        "scan_end_mass": round(sb_max / 1000.0, 4)         
                    },
                    "stats": {
                        "sr_events": int(sr_count),
                        "left_sb_events": int(left_count),
                        "right_sb_events": int(right_count)
                    }
                }
                candidates.append(rec)

            current_center += (sr_width * 0.5)

        if not candidates:
            return {"warning": f"No valid regions found above {safe_mass_floor:.0f} GeV."}
        
        # 5. Sorting and Ranking
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        
        for i, cand in enumerate(candidates):
            if i == 0:
                cand["rank_comment"] = "**BEST CHOICE**: Optimal High Mass Sensitivity."
            elif i < 3:
                cand["rank_comment"] = "Strong Candidate"
            else:
                cand["rank_comment"] = "Backup Choice"
                
        return candidates[:8] 

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Files
    parser.add_argument('--input_background', type=str, required=False, 
                        help="Path to the background data file (JSONL or NPY). Used in Training Mode (R&D).")
    parser.add_argument('--input_signal', type=str, required=False, 
                        help="Path to the signal data file (JSONL or NPY). Used in Training Mode (R&D).")
    parser.add_argument('--input_unlabeled', type=str, required=False, 
                        help="Path to the unlabeled data file (JSONL or NPY). Used in Inference Mode (Real Data).")
    
    # Physics Parameters
    parser.add_argument("--sigma_width", type=float, default=4.0, 
                        help="Width of the Signal Region (SR) in terms of resolution sigma. Default: 4.0.")
    parser.add_argument("--resolution_ratio", type=float, default=0.03, 
                        help="Estimated detector mass resolution as a fraction of mass (e.g., 0.03 for 3%).")
    
    # System Constraints
    parser.add_argument("--min_jet_pt", type=float, default=1200.0, help=(
        "AKA: Trigger Treshold. Minimum jet pT to set safe mass threshold. "
        "This is determined by the experiment trigger settings of the collider run."
    ))
    parser.add_argument("--min_events_saturation", type=int, default=40000, help=(
        "Event count threshold where statistical returns diminish. "
        "Used to cap the scoring function so it doesn't purely favor low-mass regions. "
        "Default: 40000 (approx 0.5% stat error)."
    ))
    
    args = parser.parse_args()

    results = propose_regions(
        input_background=args.input_background,
        input_signal=args.input_signal,
        input_unlabeled=args.input_unlabeled,
        resolution_ratio=args.resolution_ratio,
        sigma_width=args.sigma_width,
        min_jet_pt=args.min_jet_pt,
        min_events_saturation=args.min_events_saturation
    )

    print("<tool_result>")
    print(json.dumps(results, indent=2))
    print("</tool_result>")