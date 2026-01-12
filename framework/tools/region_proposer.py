import argparse
import numpy as np
import json
import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    data_pool_min=2000.0,
    data_pool_max=8000.0,
    scan_range_start=2500.0,
    scan_range_stop=7500.0,
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
        
        # Start Constraint: User request (scan_range_start) vs Physical Safety
        # Must be > safe floor AND > data_pool_min + anchor
        safety_floor = max(safe_mass_floor, data_pool_min + min_anchor_gev)
        
        # We start at whichever is higher: what the user asked for or what is safe
        current_sr_start = max(scan_range_start, safety_floor)

        if current_sr_start > scan_range_start:
            logger.warning(f"Adjusting requested start {scan_range_start} to {current_sr_start} GeV for safety.")

        # Stop Constraint: User request (scan_range_stop) vs Physical Safety
        # The SR end must be < scan_range_stop AND leave room for the Right Anchor
        safety_ceiling = data_pool_max - min_anchor_gev
        effective_stop = min(scan_range_stop, safety_ceiling)

        while current_sr_start + window_width <= effective_stop:
            sr_end = current_sr_start + window_width
            sr_center = (current_sr_start + sr_end) / 2
            
            # Sideband Definition:
            # We fix sidebands to the global analysis limits. 
            # The SR creates a "hole" in this global dataset.
            sb_min = data_pool_min
            sb_max = data_pool_max
            
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
                        "data_pool_min": round(sb_min / 1000.0, 4),      
                        "min_mass_signal_region": round(current_sr_start / 1000.0, 4), 
                        "max_mass_signal_region": round(sr_end / 1000.0, 4),   
                        "data_pool_max": round(sb_max / 1000.0, 4)         
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


def plot_quality_map(masses, candidates, data_pool_min, data_pool_max, trigger_threshold_pt, job_id=None):
    """
    Generates Figure: The Region Proposer Output.
    Visualizes mass histogram with a Quality Score heatmap.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})

    # 1. Top Plot: Mass Histogram
    counts, bins, _ = ax1.hist(masses, bins=100, range=(data_pool_min, data_pool_max), 
                               color='gray', alpha=0.3, label='Mass Spectrum')
    ax1.set_ylabel("Events / Bin")
    ax1.set_title("Figure 3.3: The Region Proposer Output - Quality Map", fontsize=14, fontweight='bold')
    
    # Vertical line for Trigger Safety Floor
    safe_floor = 2.0 * trigger_threshold_pt
    ax1.axvline(safe_floor, color='red', linestyle='--', label=f'Trigger Safe Floor ({int(safe_floor)} GeV)')

    # 2. Bottom Plot: Quality Heatmap
    # Create a dense grid of scores across the spectrum
    x_grid = np.linspace(data_pool_min, data_pool_max, 500)
    scores = np.zeros_like(x_grid)
    
    # Map candidate scores to the grid
    if candidates:
        max_score = max(c['quality_score'] for c in candidates)
        for c in candidates:
            m_min = c['tool_parameters']['min_mass_signal_region'] * 1000.0
            m_max = c['tool_parameters']['max_mass_signal_region'] * 1000.0
            # Fill the grid where this SR exists
            scores[(x_grid >= m_min) & (x_grid <= m_max)] = c['quality_score'] / max_score

    # Plotting the heatmap bar
    im = ax2.imshow([scores], aspect='auto', cmap='RdYlGn', 
                    extent=[data_pool_min, data_pool_max, 0, 1], interpolation='nearest')
    
    ax2.set_yticks([]) # Hide y-axis for heatmap
    ax2.set_xlabel("Invariant Mass $m_{jj}$ [GeV]")
    
    # Add text annotations for zones
    ax2.text(data_pool_min + 100, 0.5, "Bias Risk", color='black', fontweight='bold', va='center')
    ax2.text(data_pool_max - 800, 0.5, "Data Starved", color='black', fontweight='bold', va='center')

    # Colorbar and Legend
    ax1.legend()
    
    output_name = ["quality_map"]
    session_id = os.environ.get("SESSION_ID", None)
    if session_id:
        output_name.append(f"s_{session_id}")
    
    if job_id:
        output_name.append(f"job_{job_id}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_name.append(f"{timestamp}")

    output_path = f"{'_'.join(output_name)}.png"
    output_path = os.path.join("toolout/graphs/region_proposer", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visual Quality Map saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_background', type=str)
    parser.add_argument('--input_signal', type=str)
    parser.add_argument('--input_unlabeled', type=str)
    parser.add_argument('--job_id', type=str, default=None)
    
    # Scan Configuration
    parser.add_argument("--data_pool_min", type=float, default=2000.0)
    parser.add_argument("--data_pool_max", type=float, default=8000.0)
    parser.add_argument("--scan_range_start", type=float, default=2500.0)
    parser.add_argument("--scan_range_stop", type=float, default=7500.0)
    parser.add_argument("--window_width", type=float, default=400.0)
    parser.add_argument("--step_size", type=float, default=200.0)
    parser.add_argument("--trigger_threshold_pt", type=float, default=1200.0)

    args = parser.parse_args()

    job_id = args.job_id

    results = propose_regions(
        input_background=args.input_background,
        input_signal=args.input_signal,
        input_unlabeled=args.input_unlabeled,
        data_pool_min=args.data_pool_min,
        data_pool_max=args.data_pool_max,
        scan_range_start=args.scan_range_start,
        scan_range_stop=args.scan_range_stop,
        window_width=args.window_width,
        step_size=args.step_size,
        trigger_threshold_pt=args.trigger_threshold_pt
    )

    try:
        mass_data = None
        if args.input_unlabeled:
            mass_data = load_masses_from_file(args.input_unlabeled)
        elif args.input_background:
            mass_data = load_masses_from_file(args.input_background)
        
        if mass_data is not None and isinstance(results, list):
            # Ensure units match GeV
            if mass_data.max() < 100: mass_data *= 1000.0
            
            plot_quality_map(
                masses=mass_data, 
                candidates=results, 
                data_pool_min=args.data_pool_min, 
                data_pool_max=args.data_pool_max, 
                trigger_threshold_pt=args.trigger_threshold_pt,
                job_id=job_id
            )
    except Exception as e:
        logger.error(f"Could not generate plot: {e}")

    if len(results) == 0:
        print("<tool_result>")
        print("No viable signal regions found with the given parameters.")
        print("Make sure you don't have overly restrictive mass windows or insufficient data.")
        print("</tool_result>")
        sys.exit(0)

    print("<tool_result>")
    if job_id:
        print(f"Job ID: {job_id}")
    print("Following are signal region proposals to start your analysis with (ordered by quality score):")
    print(json.dumps(results, indent=2))
    print("</tool_result>")