"""
LaCATHODE Report Generator Tool
Generates an enhanced report summarizing anomaly detection results,
including statistical significance, excess factors, and top candidates.
Also produces visualizations for anomaly distributions and model validation.
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr, norm
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = ArgumentParser()
parser.add_argument("--data_file", type=str, required=True,
                    help="Path to the input data file (numpy .npy format)")
parser.add_argument("--scores_file", type=str, required=True,
                    help="Path to the anomaly scores file (numpy .npy format)")
parser.add_argument("--output_dir", type=str, required=False,
                    default="toolout/reports",
                    help="Path to save the generated report")
parser.add_argument("--print_report", action="store_true",
                    default=False,
                    help="If set, print the report to console as well")
parser.add_argument("--top_percentile", type=float, default=99.0,
                    help="Percentile threshold to define anomaly candidates")
parser.add_argument("--bin_count", type=int, default=20,
                    help="Number of bins for mass histogram")
parser.add_argument("--min_events_per_bin", type=int, default=200,
                    help="Minimum events required in a bin to calculate excess factor.")
parser.add_argument("--report_id", type=str, default=None,
                    help="Optional identifier to append to the report filename.")

# --- HELPER FUNCTIONS ---

def get_smart_bins(x_min, x_max, sr_start, sr_end, bin_count=40):
    """
    Generates bin edges that strictly respect Signal Region (SR) boundaries.
    """
    total_range = x_max - x_min
    if total_range <= 0: return np.linspace(x_min, x_max, bin_count + 1)
    
    sr_width = sr_end - sr_start
    left_width = max(0, sr_start - x_min)
    right_width = max(0, x_max - sr_end)

    # Allocate bins proportional to width
    ratio_sr = sr_width / total_range
    n_sr = int(round(bin_count * ratio_sr))
    n_sr = max(1, n_sr) if sr_width > 0 else 0
    
    remaining_bins = bin_count - n_sr
    
    # Allocate remaining to left/right
    if left_width > 0 and right_width > 0:
        ratio_left = left_width / (left_width + right_width)
        n_left = int(round(remaining_bins * ratio_left))
        n_right = remaining_bins - n_left 
        if n_left == 0: n_left = 1; n_right -= 1
        if n_right == 0: n_right = 1; n_left -= 1
    elif left_width > 0:
        n_left = remaining_bins
        n_right = 0
    else:
        n_left = 0
        n_right = remaining_bins

    edges = []
    if n_left > 0:
        edges.append(np.linspace(x_min, sr_start, n_left + 1)[:-1])
    if n_sr > 0:
        edges.append(np.linspace(sr_start, sr_end, n_sr + 1))
    if n_right > 0:
        start_idx = 1 if n_sr > 0 else 0
        edges.append(np.linspace(sr_end, x_max, n_right + 1)[start_idx:])
        
    return np.concatenate(edges)

def calculate_poisson_significance(n_obs, n_exp):
    """Vectorized Significance Calculation"""
    with np.errstate(divide='ignore', invalid='ignore'):
        q0 = 2 * (n_obs * np.log(n_obs / n_exp) - (n_obs - n_exp))
        q0 = np.where((n_exp <= 0) | (n_obs <= n_exp), 0.0, q0)
        return np.sqrt(np.maximum(0, q0))

# --- PLOTTING FUNCTIONS ---

def plot_anomaly_results(
    mass, scores, threshold, bins, save_path, 
    n_expected=None, n_expected_raw=None, top_percentile=99.0
):
    print(f"--- Generating Consistent Bump Hunt Plot ---")
    mask = scores > threshold
    selected_mass = mass[mask]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax0 = plt.subplot(gs[0]) # Left Axis (Significance)
    ax1 = plt.subplot(gs[1], sharex=ax0) # Ratio Panel

    # --- LEFT AXIS: The Physics (Signal & Model) ---
    
    # 1. Signal (Red)
    counts_sig, _, _ = ax0.hist(
        selected_mass, bins=bins, density=False, 
        color='crimson', lw=2, label=f'Anomalies (Top {100-top_percentile:.1f}%)',
        histtype='step', zorder=10
    )
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    denom_for_ratio = None
    ratio_label = "Sideband" 

    # 2. Model (Blue) - The "Correct" Background (Filtered)
    if n_expected is not None and np.sum(n_expected) > 0:
        ax0.plot(bin_centers, n_expected, color='blue', linestyle='--', 
                 label='Background Model', zorder=10)
        denom_for_ratio = n_expected
        ratio_label = "Model"

    # 3. Raw Flow (Green) - The "Diagnostic" Background (Unfiltered)
    if n_expected_raw is not None and np.sum(n_expected_raw) > 0:
        ax0.plot(bin_centers, n_expected_raw, color='forestgreen', linestyle=':', lw=1.5,
                 label='Raw Flow Shape', zorder=5)
        # If filtered model failed (is zero), use raw flow for ratio
        if denom_for_ratio is None: 
            denom_for_ratio = n_expected_raw
            ratio_label = "Raw Flow"

    # --- RIGHT AXIS: The Context (All Data) ---
    ax0_twin = ax0.twinx()
    ax0_twin.hist(
        mass, bins=bins, density=False, 
        color='gray', alpha=0.15, label='All Events (Right Scale)',
        histtype='stepfilled', edgecolor='none', zorder=1
    )
    
    # Styling
    ax0.set_ylabel("Anomaly Count", fontsize=12, color='crimson')
    ax0.tick_params(axis='y', labelcolor='crimson')
    ax0_twin.set_ylabel("Total Event Count", fontsize=12, color='gray')
    ax0_twin.tick_params(axis='y', labelcolor='gray')
    
    ax0.set_ylim(bottom=0)
    ax0_twin.set_ylim(bottom=0)
    
    # Combined Legend
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_twin.get_legend_handles_labels()
    ax0.legend(h1+h2, l1+l2, loc='upper right', frameon=True)
    
    ax0.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- RATIO PANEL ---
    if denom_for_ratio is not None:
        safe_exp = np.maximum(denom_for_ratio, 0.1)
        ratio = counts_sig / safe_exp
    else:
        # Absolute fallback
        scale = len(selected_mass) / len(mass)
        safe_bg = np.maximum(counts_sig / scale, 0.1) 
        ratio = counts_sig / (safe_bg * scale)
        ratio_label = "Scaled Bkg"

    ax1.plot(bin_centers, ratio, 'ko', markersize=4)
    ax1.axhline(1.0, color='red', linestyle='--', label=f'Ratio=1 ({ratio_label})')
    
    ax1.set_ylabel("Obs / Exp")
    ax1.set_xlabel("Invariant Mass (GeV)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    if len(ratio) > 0:
        ax1.set_ylim(0, max(2.0, np.max(ratio)*1.1))
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def plot_synthetic_validation(
    data_file, scores_file, synthetic_file, 
    save_path, top_percentile=99.0, bin_count=18
):
    print(f"Generating Synthetic Validation Plot")
    try:
        data = np.load(data_file)
        scores = np.load(scores_file)
        mass = data[:, 0]
        if np.median(mass) < 100: mass *= 1000.0 
        
        synth_data = np.load(synthetic_file)
        synth_mass = synth_data[:, 0] 
        synth_scores = synth_data[:, 1]
        if np.median(synth_mass) < 100: synth_mass *= 1000.0

    except Exception as e:
        print(f"Plotting Error: {e}")
        return

    threshold = np.percentile(scores, top_percentile)
    
    mask_real = scores > threshold
    mass_real_pass = mass[mask_real]

    mask_synth = synth_scores > threshold
    mass_synth_pass = synth_mass[mask_synth]

    # --- ROBUST FALLBACK LOGIC (Updated) ---
    norm_factor = len(mass) / len(synth_mass)
    
    observed_yield = len(mass_real_pass)
    predicted_yield = len(mass_synth_pass) * norm_factor
    
    # Threshold increased to 0.5 (50%) to catch partial overfitting
    if observed_yield > 0 and predicted_yield < 0.5 * observed_yield:
        print(f"  Validation Plot: Fallback Triggered (Pred {predicted_yield:.1f} < 50% of Obs {observed_yield}).")
        plot_data_bg = synth_mass # Use ALL events (Raw Flow)
        bg_label = "Background (Raw Flow Shape)"
        # Scale to match the COUNT of observed anomalies for comparison
        scale = observed_yield / len(plot_data_bg)
    else:
        plot_data_bg = mass_synth_pass
        bg_label = "Background (Model Filtered)"
        scale = norm_factor

    weights_bg = np.full_like(plot_data_bg, scale)
    # -----------------------------

    x_min = min(mass.min(), synth_mass.min())
    x_max = max(mass.max(), synth_mass.max())
    bins = np.linspace(x_min, x_max, bin_count)
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # A. Predicted Background
    counts_synth, _, _ = ax0.hist(
        plot_data_bg, bins=bins, weights=weights_bg,
        color='dodgerblue', alpha=0.3, label=bg_label,
        histtype='stepfilled', edgecolor='blue', lw=1
    )

    # B. Observed Anomalies
    counts_real, _ = np.histogram(mass_real_pass, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    yerr = np.sqrt(counts_real)
    
    ax0.errorbar(
        bin_centers, counts_real, yerr=yerr, 
        fmt='o', color='black', label='Observed Anomalies',
        markersize=5, capsize=2, elinewidth=1.5
    )

    ax0.set_ylabel("Events / Bin", fontsize=12)
    ax0.set_title(f"Validation: Observed vs. Predicted Background", fontsize=14, fontweight='bold')
    ax0.legend(fontsize=11)
    ax0.grid(True, which='both', linestyle='--', alpha=0.4)
    ax0.set_xticklabels([]) 

    # PANEL 2: RATIO
    safe_exp = np.maximum(counts_synth, 0.1) 
    ratio = counts_real / safe_exp
    
    ax1.plot(bin_centers, ratio, color='black', marker='o', linestyle='', markersize=4)
    ax1.axhline(1.0, color='dodgerblue', linestyle='--', lw=2)
    
    rel_err = 1.0 / np.sqrt(safe_exp)
    ax1.fill_between(bin_centers, 1 - 2*rel_err, 1 + 2*rel_err, color='dodgerblue', alpha=0.1, label=r'2$\sigma$ Band')

    ax1.set_ylabel("Obs / Pred", fontsize=10)
    ax1.set_xlabel("Invariant Mass (GeV)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 2.5) 

    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Validation plot saved to {save_path}")

def plot_full_spectrum_viz(
    data_file, scores_file, synthetic_file, 
    save_path, outer_file=None, top_percentile=99.0, bin_count=60
):
    print(f"--- Generating Full Spectrum Plot (Smart Bins & Clipped) ---")
    try:
        inner_data = np.load(data_file)
        inner_mass = inner_data[:, 0]
        if np.median(inner_mass) < 100: inner_mass *= 1000.0
        
        scores = np.load(scores_file)
        
        synth_data = np.load(synthetic_file)
        synth_mass = synth_data[:, 0]
        if np.median(synth_mass) < 100: synth_mass *= 1000.0
        
        outer_mass = np.array([])
        if outer_file and os.path.exists(outer_file):
            outer_data = np.load(outer_file)
            outer_mass = outer_data[:, 0]
            if np.median(outer_mass) < 100: outer_mass *= 1000.0
    
    except Exception as e:
        print(f"Full Spectrum Plot Error: {e}")
        return

    sr_start = inner_mass.min()
    sr_end = inner_mass.max()
    
    # Global Range
    if len(outer_mass) > 0:
        x_min = min(sr_start, outer_mass.min())
        x_max = max(sr_end, outer_mass.max())
    else:
        x_min, x_max = sr_start, sr_end

    # Data Clipping
    if len(outer_mass) > 0:
        mask_clean_outer = (outer_mass < sr_start) | (outer_mass > sr_end)
        outer_mass = outer_mass[mask_clean_outer]

    # --- CRITICAL FIX: Clip Synth BEFORE Normalizing ---
    synth_mass_clipped = synth_mass[(synth_mass >= sr_start) & (synth_mass <= sr_end)]
    
    # Normalize: Match Area of Data(SR) to Area of Model(SR)
    if len(synth_mass_clipped) > 0:
        norm_factor = len(inner_mass) / len(synth_mass_clipped)
    else:
        norm_factor = 1.0

    bins = get_smart_bins(x_min, x_max, sr_start, sr_end, bin_count)

    plt.figure(figsize=(12, 7))
    
    # 1. Sideband (Gray)
    if len(outer_mass) > 0:
        plt.hist(outer_mass, bins=bins, color='black', alpha=0.25, 
                 label='Sideband Data', histtype='stepfilled', edgecolor='none')

    # 2. Signal Region Data (Black Line)
    plt.hist(inner_mass, bins=bins, color='black', alpha=0.3, 
             label='Signal Region Data', histtype='step', lw=1, linestyle=':')

    # 3. Flow Model (Blue Fill - SR Only)
    weights = np.full_like(synth_mass_clipped, norm_factor)
    plt.hist(synth_mass_clipped, bins=bins, weights=weights, color='dodgerblue', alpha=0.4,
             label='Flow Model (SR)', histtype='stepfilled', edgecolor='blue')

    # 4. Anomalies (Red Fill)
    threshold = np.percentile(scores, top_percentile)
    mass_anom = inner_mass[scores > threshold]
    plt.hist(mass_anom, bins=bins, color='crimson', alpha=0.7, 
             label=f'Anomalies (Top {100-top_percentile}%)', histtype='stepfilled', edgecolor='black')

    plt.yscale('log')
    plt.title(f"Full Physics Picture: Sideband - Signal Region - Sideband")
    plt.xlabel("Invariant Mass (GeV)")
    plt.ylabel("Events (Log Scale)")
    
    plt.axvline(sr_start, color='black', linestyle='--', alpha=0.5)
    plt.axvline(sr_end, color='black', linestyle='--', alpha=0.5)

    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Full spectrum plot saved to {save_path}")


# --- MAIN LOGIC ---

def generate_report(**args):
    data_file = args['data_file']
    scores_file = args['scores_file']
    
    name_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.get("output_dir", "toolout/reports")
    print_report = args.get("print_report", False)

    session_id = os.environ.get("FRAMEWORK_SESSION_ID")
    if session_id:
        output_dir = os.path.join(output_dir, session_id)
    
    if args.get("report_id"):
        output_dir = os.path.join(output_dir, f"report_{args['report_id']}")

    if not output_dir and not print_report:
        print("No output method specified. Exiting.")
        return
    
    output_file = os.path.join(output_dir, f"lacathode_report_{name_timestamp}.md")

    top_percentile = args.get("top_percentile", 99.0)
    bin_count = args.get("bin_count", 20)

    print(f"Generating Enhanced Report from {data_file}...")
    
    if not os.path.exists(data_file) or not os.path.exists(scores_file):
        print("Error: Files not found.")
        return

    # Load Data
    data = np.load(data_file)
    scores = np.load(scores_file)
    
    # Sync lengths
    n = min(len(data), len(scores))
    data, scores = data[:n], scores[:n]

    # Unit Correction
    mass = data[:, 0]
    if np.median(mass) < 100: 
        mass *= 1000.0
        data[:, 7] *= 1000.0; data[:, 18] *= 1000.0

    # Define Signal (Top X%)
    threshold = np.percentile(scores, top_percentile)
    is_signal = scores > threshold
    sig_mass = mass[is_signal]

    # --- BINS & BACKGROUND ESTIMATION ---
    sr_start = mass.min()
    sr_end = mass.max()
    
    # Use unified bins
    bins = get_smart_bins(sr_start, sr_end, sr_start, sr_end, bin_count)

    hist_all, _ = np.histogram(mass, bins=bins)
    hist_sig, _ = np.histogram(sig_mass, bins=bins) # n_observed

    # Load Synthetic Background & Apply Smart Fallback
    synthetic_file = args['scores_file'].replace('.npy', '_synthetic.npy')
    n_expected = np.zeros_like(hist_sig, dtype=float)
    n_expected_raw = None # For plotting only
    fallback_triggered = False
    
    if os.path.exists(synthetic_file):
        try:
            print(f"Loading synthetic background from {synthetic_file}...")
            synth_data = np.load(synthetic_file)
            synth_masses = synth_data[:, 0]
            synth_scores = synth_data[:, 1]
            
            # Unit Correction
            if np.median(synth_masses) < 100: 
                synth_masses *= 1000.0
            
            # METHOD A: Classifier Filtered (The Standard)
            synth_passing = synth_masses[synth_scores > threshold]
            hist_synth_filtered, _ = np.histogram(synth_passing, bins=bins)
            norm_factor = len(mass) / len(synth_data)
            n_expected = hist_synth_filtered * norm_factor

            # CHECK FOR OVERFITTING / FALLBACK
            obs_count = np.sum(hist_sig)
            exp_count = np.sum(n_expected)
            
            # If Exp is suspiciously low (< 20% of Obs) or zero
            if obs_count > 0 and exp_count < 0.2 * obs_count:
                print(f"WARNING: Classifier Overfitting Detected (Exp={exp_count:.1f} vs Obs={obs_count}).")
                print("-> Switching to 'Raw Flow' (Shape-Only) comparison.")

                fallback_triggered = True
                
                # Load Raw Shape
                hist_synth_raw, _ = np.histogram(synth_masses, bins=bins)
                
                # Scale Raw Shape to match Observed Count
                raw_count = np.sum(hist_synth_raw)
                if raw_count > 0:
                    scale_factor = obs_count / raw_count
                    n_expected = hist_synth_raw * scale_factor
                    
            # Save Raw Flow anyway for visual diagnostic (Green Line)
            hist_synth_raw, _ = np.histogram(synth_masses, bins=bins)
            raw_count = np.sum(hist_synth_raw)
            if raw_count > 0:
                scale_diag = np.sum(hist_sig) / raw_count
                n_expected_raw = hist_synth_raw * scale_diag

        except Exception as e:
            print(f"Synthetic Data Calculation Error: {e}")
            
    # Absolute Fallback: Sideband Interpolation
    if np.sum(n_expected) == 0:
        n_expected = (np.roll(hist_sig, 1) + np.roll(hist_sig, -1)) / 2.0
        n_expected[0] = hist_sig[1]
        n_expected[-1] = hist_sig[-2]

    n_expected = np.maximum(n_expected, 0.1)

    # --- STATISTICS ---
    significances = calculate_poisson_significance(hist_sig, n_expected)
    excess_factor = hist_sig / n_expected
    
    # Filter low stats
    MIN_EVENTS_THRESHOLD = args.get("min_events_per_bin", 500)
    low_stats_mask = hist_all < MIN_EVENTS_THRESHOLD
    excess_factor[low_stats_mask] = 0.0 
    significances[low_stats_mask] = 0.0

    max_excess_idx = np.argmax(significances)
    max_sig_val = significances[max_excess_idx]
    max_excess_val = excess_factor[max_excess_idx]
    p_value = norm.sf(max_sig_val)
    
    hotspot_min = bins[max_excess_idx]
    hotspot_max = bins[max_excess_idx+1]
    hotspot_center = (hotspot_min + hotspot_max) / 2
    events_in_hotspot = hist_sig[max_excess_idx]

    corr, _ = pearsonr(mass, scores)
    bias_warning = "(WARN: High Bias)" if abs(corr) > 0.1 else "(OK: Low Bias)"

    # --- TOP CANDIDATES ---
    hotspot_mask = (mass >= hotspot_min) & (mass < hotspot_max)
    hotspot_indices = np.where(hotspot_mask)[0]
    hotspot_local_scores = scores[hotspot_indices]
    local_sort_idx = np.argsort(hotspot_local_scores)[-5:][::-1]
    top_hotspot_indices = hotspot_indices[local_sort_idx]

    # This part especially written by an LLM agent, they should understand each other shouldn't they? - Anıl
    # --- GENERATE MARKDOWN REPORT ---
    report = []
    report.append("# Anomaly Analysis Report")
    report.append("A summary of LaCATHODE model findings on the dataset.\n")
    
    report.append("## 0. Executive Summary")
    
    # --- CONTEXT-AWARE DIAGNOSTICS FOR LLM AGENTS ---
    if fallback_triggered:
        report.append(f"- **Analysis Mode:** SHAPE-ONLY (Rate estimation unreliable)")
        
        if max_sig_val > 3.0:
            # Case A: We rejected background but found a huge bump -> Likely Signal
            report.append(f"- **Status:** POTENTIAL DISCOVERY (Significance {max_sig_val:.2f}$\\sigma$)")
            report.append("  - **Interpretation:** The classifier rejected the background but kept a massive excess of data.")
            report.append("  - **Action:** Verification recommended, but this is a strong positive signal.")
        else:
            # Case B: We rejected background and found nothing -> Likely Broken Model
            report.append(f"- **Status:** INCONCLUSIVE (Significance {max_sig_val:.2f}$\\sigma$)")
            report.append("  - **Interpretation:** The classifier rejected the background and found no significant structure.")
            report.append("  - **Action:** This indicates model failure/overfitting. Retraining is recommended.")
            
    else:
        # Standard Operation (Model passed quality checks)
        # Lowered threshold to 2.0 so 4B LLM notices "Moderate" signals (like your 2.53 sigma)
        status_icon = "" if max_sig_val > 2.0 else ""
        if max_sig_val > 5.0: status_icon = "" # Discovery threshold
        
        report.append(f"- **Analysis Mode:** STANDARD (Rate + Shape)")
        report.append(f"- **Status:** {status_icon} Valid Result (Significance {max_sig_val:.2f}$\\sigma$)")
        
        if max_sig_val > 3.0:
             report.append("  - **Interpretation:** Strong evidence of anomaly found.")
        elif max_sig_val > 2.0:
             report.append("  - **Interpretation:** Moderate evidence of anomaly found. Worth investigating.")

    report.append(f"- **Max Significance:** {max_sig_val:.2f} sigma (Confidence)")
    report.append(f"- **Null Hypothesis p-value:** {p_value:.2e}")
    report.append(f"- **Max Excess:** {max_excess_val:.2f}x baseline")
    report.append(f"- **Hotspot Location:** {hotspot_min:.0f} - {hotspot_max:.0f} GeV (Center: {hotspot_center:.0f} GeV)")
    report.append(f"- **Events in Hotspot:** {events_in_hotspot} anomalies")
    report.append(f"- **Global Threshold:** Top {100-top_percentile}% (Score > {threshold:.4f})")
    report.append(f"- **Mass-Score Correlation:** {corr:.3f} {bias_warning}\n")
    
    report.append("## 1. Mass Spectrum Scan")
    df_hist = pd.DataFrame({
        'Bin_Start': bins[:-1].astype(int),
        'Bin_End': bins[1:].astype(int),
        'Observed': hist_sig.astype(int),
        'Expected': np.round(n_expected, 1),
        'Excess': np.round(excess_factor, 3),
        'Sigma': np.round(significances, 3)
    })
    report.append(df_hist.to_markdown(index=False))
    
    report.append("\n\n## 2. Top 3 Highest Scored Events")
    top_indices = np.argsort(scores)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        evt = data[idx]
        report.append(f"### Candidate #{i+1} (Mass: {evt[0]*1000 if evt[0]<100 else evt[0]:.0f} GeV)")
        report.append(f"- **Score:** {scores[idx]:.4f}")
        report.append(f"- **Jet 1:** Mass={evt[7]:.0f}, pT={evt[9]:.0f}, Tau21={evt[14]:.3f}")
        report.append(f"- **Jet 2:** Mass={evt[18]:.0f}, pT={evt[20]:.0f}, Tau21={evt[25]:.3f}")
        report.append(f"- **dR:** {evt[2]:.3f}\n")

    report.append("## 3. Top 5 Hotspot Candidates")
    for i, idx in enumerate(top_hotspot_indices):
        evt = data[idx]
        report.append(f"### Hotspot Candidate #{i+1} (Mass: {evt[0]*1000 if evt[0]<100 else evt[0]:.0f} GeV)")
        report.append(f"- **Score:** {scores[idx]:.4f}")
        report.append(f"- **dR:** {evt[2]:.3f}\n")

    # Save Report
    if print_report:
        print("<tool_result>")
        print("\n".join(report))
        print("</tool_result>")

    if output_dir:
        dir_name = os.path.dirname(output_file)
        if dir_name and not os.path.exists(dir_name):    
            os.makedirs(dir_name, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(report))
        if not print_report:
            print(f"Enhanced Report saved to {output_file}")

    # --- GENERATE PLOTS ---
    output_without_dir = os.path.basename(output_file)
    plot_path = output_without_dir.replace('.md', '_bump_hunt.png')
    plot_path = os.path.join(output_dir, plot_path)

    if output_dir:
        # 1. Main Bump Hunt Plot (Using the robust n_expected we just calculated)
        plot_anomaly_results(
            mass=mass, 
            scores=scores, 
            threshold=threshold, 
            bins=bins,           
            save_path=plot_path,
            n_expected=n_expected,
            n_expected_raw=n_expected_raw, # Pass the green line data
            top_percentile=top_percentile
        )

        # 2. Synthetic Validation & Full Spectrum (Auxiliary Plots)
        if os.path.exists(synthetic_file):
            try:
                # Validation Plot
                validation_plot_path = plot_path.replace('_bump_hunt.png', '_validation_viz.png')
                plot_synthetic_validation(
                    data_file=data_file,
                    scores_file=scores_file,
                    synthetic_file=synthetic_file,
                    save_path=validation_plot_path,
                    top_percentile=top_percentile,
                    bin_count=bin_count
                )

                # Full Spectrum Plot
                full_viz_path = plot_path.replace('_bump_hunt.png', '_full_spectrum.png')
                outer_file_path = data_file.replace('inner', 'outer')
                if os.path.exists(outer_file_path):
                    plot_full_spectrum_viz(
                        data_file=data_file,
                        scores_file=scores_file,
                        synthetic_file=synthetic_file,
                        save_path=full_viz_path,
                        outer_file=outer_file_path,
                        top_percentile=top_percentile,
                        bin_count=60 
                    )
            except Exception as e:
                print(f"Error generating aux plots: {e}")

if __name__ == "__main__":
    args = parser.parse_args()
    generate_report(**vars(args))