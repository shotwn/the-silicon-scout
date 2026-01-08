import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_file", type=str, required=True,
                    help="Path to the input data file (numpy .npy format)")
parser.add_argument("--scores_file", type=str, required=True,
                    help="Path to the anomaly scores file (numpy .npy format)")
parser.add_argument("--output_file", type=str, default="llm_enhanced_report.txt",
                    help="Path to save the generated report")
parser.add_argument("--print_report", action="store_true",
                    default=False,
                    help="If set, print the report to console as well")
parser.add_argument("--top_percentile", type=float, default=99.0,
                    help="Percentile threshold to define anomaly candidates")
parser.add_argument("--bin_count", type=int, default=18,
                    help="Number of bins for mass histogram")
parser.add_argument("--min_events_per_bin", type=int, default=200,
                    help="Minimum events required in a bin to calculate excess factor (filters noise).")

def generate_report(**args):
    data_file = args['data_file']
    scores_file = args['scores_file']
    
    output_file = args.get("output_file", "llm_enhanced_report.txt")
    print_report = args.get("print_report", False)

    if not output_file and not print_report:
        print("No output method specified (neither file nor print). Exiting.")
        return

    top_percentile = args.get("top_percentile", 99.0)
    bin_count = args.get("bin_count", 18)

    print(f"Generating Enhanced Report from {data_file}...")
    
    if not os.path.exists(data_file) or not os.path.exists(scores_file):
        print("Error: Files not found.")
        return

    # 1. Load Data
    data = np.load(data_file)
    scores = np.load(scores_file)
    
    # Sync lengths
    n = min(len(data), len(scores))
    data, scores = data[:n], scores[:n]

    # Unit Correction
    mass = data[:, 0]
    if mass.max() < 100: 
        mass *= 1000.0
        data[:, 7] *= 1000.0; data[:, 18] *= 1000.0

    # 2. Define Signal (Top X%)
    threshold = np.percentile(scores, top_percentile)
    is_signal = scores > threshold
    sig_mass = mass[is_signal]

    # 3. Calculate Histogram & Excess
    bins = np.linspace(mass.min(), mass.max(), bin_count) # 18 bins
    hist_all, _ = np.histogram(mass, bins=bins)
    hist_sig, _ = np.histogram(sig_mass, bins=bins)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = hist_sig / hist_all
        density = np.nan_to_num(density, nan=0.0)
    
    excess_factor = density / 0.01 # 1.0 = Normal, >1.0 = Excess

    # Filter out bins with low statistics to prevent "100.0" artifacts
    # If a bin has fewer than 500 events, we consider the excess calculation unreliable.
    MIN_EVENTS_THRESHOLD = args.get("min_events_per_bin", 500)
    low_stats_mask = hist_all < MIN_EVENTS_THRESHOLD
    excess_factor[low_stats_mask] = 0.0 

    # Calculate Max Excess Info
    max_excess_idx = np.argmax(excess_factor)
    max_excess_val = excess_factor[max_excess_idx]
    
    # Get the mass range where this excess happened
    hotspot_min = bins[max_excess_idx]
    hotspot_max = bins[max_excess_idx+1]
    hotspot_center = (hotspot_min + hotspot_max) / 2
    
    # Count events in the hotspot to judge "significance"
    events_in_hotspot = hist_sig[max_excess_idx]

    # Calculates if the model is biased (learning mass instead of physics)
    corr, _ = pearsonr(mass, scores)
    bias_warning = "(WARN: High Bias)" if abs(corr) > 0.1 else "(OK: Low Bias)"

    # --- CRITICAL CHANGE: Filter for events INSIDE the Hotspot ---
    # 1. Identify all events that fall into the hotspot mass bin
    hotspot_mask = (mass >= hotspot_min) & (mass < hotspot_max)
    hotspot_indices = np.where(hotspot_mask)[0]
    
    # 2. Get the scores for these specific events
    hotspot_local_scores = scores[hotspot_indices]
    
    # 3. Sort them to find the top 5 *local* candidates
    # argsort gives indices relative to the small array, so we map them back
    local_sort_idx = np.argsort(hotspot_local_scores)[-5:][::-1]
    top_hotspot_indices = hotspot_indices[local_sort_idx]

    # 4. Generate Report
    report = []
    report.append("# Anomaly Analysis Report")
    report.append("A summary of LaCATHODE model findings on the dataset.\n")
    
    # --- SECTION 0: EXECUTIVE SUMMARY (The new stuff) ---
    report.append("## 0. Executive Summary")
    report.append(f"- **Max Excess Found:** {max_excess_val:.2f}x baseline (Expected: 1.0x)")
    report.append(f"- **Hotspot Location:** {hotspot_min:.0f} - {hotspot_max:.0f} GeV (Center: {hotspot_center:.0f} GeV)")
    report.append(f"- **Event Count at Hotspot:** {events_in_hotspot} anomaly candidates")
    report.append(f"- **Global Threshold:** Top {100-top_percentile}% (Score > {threshold:.4f})")
    report.append(f"- **Mass-Score Correlation:** {corr:.3f} {bias_warning}\n")
    
    # --- SECTION 1: BUMP HUNT DATA ---
    report.append("## 1. Mass Spectrum Scan")
    report.append("Distribution of anomaly candidates across the mass range.\n")
    
    df_hist = pd.DataFrame({
        'Bin_Start': bins[:-1].astype(int),
        'Bin_End': bins[1:].astype(int),
        'Total_Evts': hist_all,
        'Anomaly_Evts': hist_sig,
        'Excess_Factor': np.round(excess_factor, 2)
    })
    report.append(df_hist.to_markdown(index=False))
    
    # --- SECTION 2: TOP CANDIDATES ---
    report.append("\n\n## 2. Top 3 Highest Scored Events")
    report.append("Detailed kinematics of the most anomalous events found.\n")
    
    top_indices = np.argsort(scores)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        evt = data[idx]
        report.append(f"### Candidate #{i+1} (Mass: {evt[0]*1000 if evt[0]<100 else evt[0]:.0f} GeV)")
        report.append(f"- **Score:** {scores[idx]:.4f}")
        report.append(f"- **Jet 1:** Mass={evt[7]:.0f}, pT={evt[9]:.0f}, Tau21={evt[14]:.3f}")
        report.append(f"- **Jet 2:** Mass={evt[18]:.0f}, pT={evt[20]:.0f}, Tau21={evt[25]:.3f}")
        report.append(f"- **dR:** {evt[2]:.3f}\n")

    report.append("## 3. Top 5 Hotspot Candidates")
    report.append(f"Most anomalous events located within the identified hotspot ({hotspot_min:.0f} - {hotspot_max:.0f} GeV).\n")

    for i, idx in enumerate(top_hotspot_indices):
        evt = data[idx]
        report.append(f"### Hotspot Candidate #{i+1} (Mass: {evt[0]*1000 if evt[0]<100 else evt[0]:.0f} GeV)")
        report.append(f"- **Score:** {scores[idx]:.4f}")
        report.append(f"- **Jet 1:** Mass={evt[7]:.0f}, pT={evt[9]:.0f}, Tau21={evt[14]:.3f}")
        report.append(f"- **Jet 2:** Mass={evt[18]:.0f}, pT={evt[20]:.0f}, Tau21={evt[25]:.3f}")
        report.append(f"- **dR:** {evt[2]:.3f}\n")

    # Save
    if print_report:
        print("<tool_result>")
        print("\n".join(report))
        print("</tool_result>")

    if output_file:    
        with open(output_file, "w") as f:
            f.write("\n".join(report))
        
    if not print_report:
        print("<tool_result>")
        print(f"Enhanced Report saved to {output_file} relative to current working directory.")
        print("</tool_result>")
    
    print(f"Enhanced Report saved to {output_file} relative to current working directory.")

if __name__ == "__main__":
    args = parser.parse_args()
    generate_report(**vars(args))