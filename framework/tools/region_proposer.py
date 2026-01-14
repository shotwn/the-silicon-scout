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
toolout_warnings = []

def log_warnings(warning_msg):
    toolout_warnings.append(warning_msg)
    logger.warning(warning_msg)

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

def reshape_datapool_if_needed(scan_range_start, scan_range_stop, data_pool_min, data_pool_max):
    """
    Ensures that the scanning range fits within the data pool.
    Adjusts the data pool if necessary.
    """
    adjusted_data_pool_min = data_pool_min
    adjusted_data_pool_max = data_pool_max
    margin = 500.0  # 0.5 TeV margin

    if scan_range_start - margin < data_pool_min:
        adjusted_data_pool_min = scan_range_start - margin
        log_warnings(f"Adjusting data pool min from {data_pool_min} to {adjusted_data_pool_min} GeV to fit scan range.")

    if scan_range_stop + margin > data_pool_max:
        adjusted_data_pool_max = scan_range_stop + margin
        log_warnings(f"Adjusting data pool max from {data_pool_max} to {adjusted_data_pool_max} GeV to fit scan range.")
    return adjusted_data_pool_min, adjusted_data_pool_max

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
    """
    
    # Load Data
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

        # Reshape Data Pool if Needed
        data_pool_min, data_pool_max = reshape_datapool_if_needed(
            scan_range_start, scan_range_stop, data_pool_min, data_pool_max
        )

        # Scanning Routine
        candidates = []

        # The invariant mass turn-on is roughly 2x the jet pT trigger threshold.
        safe_mass_floor = 2.0 * trigger_threshold_pt
        
        # Enforce Minimum Anchor Width (0.5 TeV = 500 GeV)
        min_anchor_gev = 500.0
        
        # Start Constraint
        safety_floor = max(safe_mass_floor, data_pool_min + min_anchor_gev)
        current_sr_start = max(scan_range_start, safety_floor)

        if current_sr_start > scan_range_start:
            log_warnings(f"Adjusting requested start {scan_range_start} to {current_sr_start} GeV for safety."
                           " Start should be at least 2x trigger threshold and allow for anchor regions.")

        # Stop Constraint
        safety_ceiling = data_pool_max - min_anchor_gev
        effective_stop = min(scan_range_stop, safety_ceiling)

        while current_sr_start + window_width <= effective_stop:
            sr_end = current_sr_start + window_width
            sr_center = (current_sr_start + sr_end) / 2
            
            sb_min = data_pool_min
            sb_max = data_pool_max
            
            # Event Counting
            left_mask = (masses >= sb_min) & (masses < current_sr_start)
            sr_mask = (masses >= current_sr_start) & (masses < sr_end)
            right_mask = (masses >= sr_end) & (masses < sb_max)
            
            left_count = np.sum(left_mask)
            sr_count = np.sum(sr_mask)
            right_count = np.sum(right_mask)

            # Scoring Logic: Bottleneck Statistic
            bottleneck_stat = min(left_count, right_count, sr_count)
            
            # Filter unstable regions
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

        # Output: Return sorted by robustness
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return candidates[:15]

    except Exception as e:
        return {"error": str(e)}

def standard_sliding_window_scan(
    input_background,
    input_signal,
    input_unlabeled,
    data_pool_min,
    data_pool_max,
    scan_range_start,
    scan_range_stop,
    window_width,
    step_size,
    trigger_threshold_pt
):
    try:
        if input_unlabeled:
            masses = load_masses_from_file(input_unlabeled)
        elif input_background and input_signal:
            bg_masses = load_masses_from_file(input_background)
            sig_masses = load_masses_from_file(input_signal)
            masses = np.concatenate([bg_masses, sig_masses])
        else:
            return {"error": "No input files provided."}

        if masses is None or len(masses) == 0:
            return {"error": "No events found."}

        # Standardization: Ensure GeV units for consistency with trigger thresholds
        if masses.max() < 100:
            masses *= 1000.0
    except Exception as e:
        return {"error": str(e)}
    
    # Regular sliding window without rating or filtering
    scan_results = []

    # Reshape Data Pool if Needed
    data_pool_min, data_pool_max = reshape_datapool_if_needed(
        scan_range_start, scan_range_stop, data_pool_min, data_pool_max
    )
    
    # Start from 2 * trigger threshold
    safe_mass_floor = 2 * trigger_threshold_pt
    current_sr_start = max(scan_range_start, safe_mass_floor)
    effective_stop = min(scan_range_stop, data_pool_max - window_width)

    while current_sr_start + window_width <= effective_stop:
        sr_end = current_sr_start + window_width
        sr_center = (current_sr_start + sr_end) / 2
        
        sb_min = data_pool_min
        sb_max = data_pool_max
        
        # Event Counting
        left_mask = (masses >= sb_min) & (masses < current_sr_start)
        sr_mask = (masses >= current_sr_start) & (masses < sr_end)
        right_mask = (masses >= sr_end) & (masses < sb_max)
        
        left_count = np.sum(left_mask)
        sr_count = np.sum(sr_mask)
        right_count = np.sum(right_mask)

        rec = {
            "focus_mass_gev": int(sr_center),
            "left_sb_events": int(left_count),
            "sr_events": int(sr_count),
            "right_sb_events": int(right_count),
            "tool_parameters": {
                "data_pool_min": round(sb_min / 1000.0, 4),      
                "min_mass_signal_region": round(current_sr_start / 1000.0, 4), 
                "max_mass_signal_region": round(sr_end / 1000.0, 4),   
                "data_pool_max": round(sb_max / 1000.0, 4)         
            }
        }
        scan_results.append(rec)

        current_sr_start += step_size

    return scan_results

def plot_quality_map(masses, candidates, data_pool_min, data_pool_max, 
                     trigger_threshold_pt, window_width, step_size, job_id=None):
    """
    Generates Figure: The Region Proposer Output - Quality Map.
    Dashboard style with Interpretation Guide.
    """
    
    # Grid Layout: 
    # Left Column (70%): Mass Histogram (Top) + Heatmap (Bottom)
    # Right Column (30%): Info Panel
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[5, 1], 
                          wspace=0.1, hspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0]) # Histogram
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # Heatmap
    ax_info = fig.add_subplot(gs[:, 1]) # Right Panel
    ax_info.axis('off')

    # --- 1. Top Plot: Mass Histogram ---
    counts, bins, _ = ax1.hist(masses, bins=100, range=(data_pool_min, data_pool_max), 
                               color='darkgray', alpha=0.4, label='Available Data', histtype='stepfilled')
    ax1.set_ylabel("Events / Bin", fontsize=12)
    ax1.set_title("Region Proposer: Mass Spectrum & Scan Quality", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Trigger Floor Line
    safe_floor = 2.0 * trigger_threshold_pt
    ax1.axvline(safe_floor, color='crimson', linestyle='--', linewidth=2, label=f'Trigger Floor ({int(safe_floor)} GeV)')
    
    # Scan Range Highlights
    if candidates:
        # Highlight the best candidate window
        best = candidates[0]['tool_parameters']
        best_min = best['min_mass_signal_region'] * 1000
        best_max = best['max_mass_signal_region'] * 1000
        ax1.axvspan(best_min, best_max, color='limegreen', alpha=0.2, label='Best Search Region')

    ax1.legend(loc='upper right')

    # --- 2. Bottom Plot: Quality Heatmap ---
    x_grid = np.linspace(data_pool_min, data_pool_max, 500)
    scores = np.zeros_like(x_grid)
    
    if candidates:
        max_score = max(c['quality_score'] for c in candidates)
        for c in candidates:
            m_min = c['tool_parameters']['min_mass_signal_region'] * 1000.0
            m_max = c['tool_parameters']['max_mass_signal_region'] * 1000.0
            # Fill grid
            scores[(x_grid >= m_min) & (x_grid <= m_max)] = c['quality_score'] / max_score

    # Plot Heatmap
    # Using 'RdYlGn' (Red=Bad, Green=Good)
    ax2.imshow([scores], aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
               extent=[data_pool_min, data_pool_max, 0, 1], interpolation='nearest')
    
    ax2.set_yticks([]) 
    ax2.set_xlabel("Invariant Mass $m_{jj}$ [GeV]", fontsize=12)
    
    # Annotations on Heatmap
    ax2.text(data_pool_min + 200, 0.5, "TRIGGER BIAS RISK", color='black', fontweight='bold', va='center', fontsize=9, alpha=0.7)
    ax2.text(data_pool_max - 1500, 0.5, "LOW STATISTICS", color='black', fontweight='bold', va='center', fontsize=9, alpha=0.7)

    # --- 3. Info Panel (Right Side) ---
    info_text = (
        r"$\bf{INTERPRETATION\ GUIDE}$" + "\n\n"
        r"$\bf{What\ is\ this?}$" + "\n"
        "A stability map for Anomaly Detection.\n"
        "We scan the spectrum to find regions where\n"
        "background estimation is statistically robust.\n\n"
        r"$\bf{The\ Heatmap\ (Bottom\ Bar)}$" + "\n"
        "Shows the 'Quality Score' for placing a\n"
        "Signal Region (SR) at that mass.\n\n"
        r"$\bullet\ \bf{Green\ Zone:}$" + " Optimal.\n"
        "Sufficient data in Left SB, SR, and Right SB.\n"
        "Safe for LaCATHODE training.\n\n"
        r"$\bullet\ \bf{Red/Yellow\ Zone:}$" + " Unstable.\n"
        "Low event counts in one of the regions.\n"
        "- Low Mass: Trigger turn-on effects.\n"
        "- High Mass: Kinematic tail (data starvation).\n\n"
        r"$\bf{Recommendation}$" + "\n"
        "Select a window from the Greenest area.\n"
        "The 'Best Search Region' (Green band) is\n"
        "automatically highlighted."
    )
    ax_info.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top', linespacing=1.6)

    # --- 4. Debug Watermark ---
    debug_text = (
        f"RUN CONFIGURATION:\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"Scan Window: {int(window_width)} GeV\n"
        f"Scan Step: {int(step_size)} GeV\n"
        f"Trigger Threshold: {int(trigger_threshold_pt)} GeV\n"
        f"Job ID: {job_id if job_id else 'N/A'}"
    )
    plt.figtext(0.98, 0.02, debug_text, ha='right', va='bottom', 
                fontsize=9, color='dimgray', family='monospace', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # Save
    output_name = ["quality_map"]
    session_id = os.environ.get("FRAMEWORK_SESSION_ID", None)
    if session_id:
        output_name.append(f"s_{session_id}")
    if job_id:
        output_name.append(f"job_{job_id}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_name.append(f"{timestamp}")

    output_path = f"{'_'.join(output_name)}.png"
    output_path = os.path.join("toolout/graphs/region_proposer", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
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

    basic_sliding_window_results = standard_sliding_window_scan(
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
                window_width=args.window_width,  # Passed for debug info
                step_size=args.step_size,        # Passed for debug info
                job_id=args.job_id
            )
    except Exception as e:
        logger.error(f"Could not generate plot: {e}")

    print("<tool_result>")
    for warning in toolout_warnings:
        print(f"WARNING: {warning}")

    if len(results) == 0 or ("error" in results if isinstance(results, dict) else False):
        print("No viable signal regions found in smart signal region proposer with the given parameters.")
        print("Make sure you don't have overly restrictive mass windows or insufficient data.")

    if len(results) > 0 and isinstance(results, list):
        if args.job_id:
            print(f"Job ID: {args.job_id}")
        print("Following are signal region proposals which are good candidates to start your analysis with (ordered by quality score):")
        print("```json")
        print(json.dumps(results, indent=2))
        print("```")

    if isinstance(basic_sliding_window_results, list) and len(basic_sliding_window_results) > 0:
        print("\nFull Sliding Window Scan Results:")
        print("```json")
        print(json.dumps(basic_sliding_window_results, indent=2))
        print("```")

    print("IMPORTANT: Results are just heuristic-based suggestions. You can still scan other regions based on your analysis needs.")
    print("</tool_result>")