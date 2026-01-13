"""
Standalone Isolation Forest Tool for FastJet Events
--------------------------------------------------
This script implements an Isolation Forest anomaly detection algorithm
tailored for FastJet JSONL event data. It supports both R&D benchmarking
and real data inference modes.

It is build upon sklearn's IsolationForest and includes detailed reporting
and visualization features to aid in interpreting the results.

Code is a little bit of spaghetti, due time constraints, but it works!
--------------------------------------------------
"""
import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
parser = argparse.ArgumentParser(description="Run Standalone Isolation Forest on FastJet Events")

# R&D Mode Inputs (Labeled)
parser.add_argument('--input_background', type=str, required=False, 
                    help="Path to background_events.jsonl (for R&D/Benchmarking)")
parser.add_argument('--input_signal', type=str, required=False, 
                    help="Path to signal_events.jsonl (for R&D/Benchmarking)")

# Inference Mode Inputs (Unlabeled)
parser.add_argument('--input_unlabeled', type=str, required=False, 
                    help="Path to unlabeled_events.jsonl (Real Data Mode)")

# Tracking
parser.add_argument('--job_id', type=str, default=None, help="Job ID for tracking outputs")

# Hyperparameters
parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
parser.add_argument("--contamination", type=str, default="auto", help="Expected anomaly fraction")
parser.add_argument("--plot", action="store_true", help="Generate score distribution plot", default=True)

parser.add_argument("--region_start", type=float, default=None, help="Start of signal region focus (TeV)")
parser.add_argument("--region_end", type=float, default=None, help="End of signal region focus (TeV)")

args = parser.parse_args()

def extract_features(event):
    """
    Extracts features from the nested FastJet JSONL format.
    """
    jets = event.get("jets", [])
    
    # --- FIX: The feature vector below has 11 elements, so the fallback must be 11. ---
    if len(jets) < 2: 
        return [0.0] * 11  # <--- CHANGED FROM 12 TO 11

    j1 = jets[0]
    j2 = jets[1]

    # --- Helper: Convert Cartesian (px, py, pz) to Kinematics (pT, eta) ---
    def get_kinematics(jet_dict):
        px = jet_dict.get("px", 0.0)
        py = jet_dict.get("py", 0.0)
        pz = jet_dict.get("pz", 0.0)
        m  = jet_dict.get("m", 0.0)
        
        # Calculate pT
        pt = np.sqrt(px**2 + py**2)
        
        # Calculate Eta
        p = np.sqrt(px**2 + py**2 + pz**2)
        if p - pz == 0 or p + pz == 0: 
            eta = 0.0
        else:
            eta = 0.5 * np.log((p + pz) / (p - pz))
            
        return pt, eta, m

    j1_pt, j1_eta, j1_m = get_kinematics(j1)
    j2_pt, j2_eta, j2_m = get_kinematics(j2)

    # --- Helper: Calculate N-subjettiness Ratio (tau21) ---
    def get_tau21(jet_dict):
        tau1 = jet_dict.get("tau_1", 1e-9) 
        tau2 = jet_dict.get("tau_2", 0.0)
        return tau2 / tau1 if tau1 > 0 else 0.0

    j1_tau21 = get_tau21(j1)
    j2_tau21 = get_tau21(j2)

    # --- Feature Vector Construction ---
    features = [
        float(event.get("m_jj", 0.0)),      # 1. Invariant Mass
        float(abs(j1_m - j2_m)),            # 2. Mass Difference
        float(event.get("dR", 0.0)),        # 3. Delta R
        
        # Jet 1
        float(j1_pt),                       # 4
        float(j1_eta),                      # 5
        float(j1_m),                        # 6
        float(j1_tau21),                    # 7
        
        # Jet 2
        float(j2_pt),                       # 8
        float(j2_eta),                      # 9
        float(j2_m),                        # 10
        float(j2_tau21)                     # 11
    ]
    
    return features

def load_jsonl(filepath, limit=None):
    """Loads a JSONL file into a numpy feature matrix."""
    if not filepath or not os.path.exists(filepath):
        return None
    
    data_list = []
    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                if not line.strip(): continue
                
                event = json.loads(line)
                data_list.append(extract_features(event))
                
        return np.array(data_list)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    print(f"--- Starting Isolation Forest Analysis ---")
    
    X = None
    y = None
    
    if args.input_unlabeled:
        print("Mode: Inference (Unlabeled Data)")
        X = load_jsonl(args.input_unlabeled)
    elif args.input_background and args.input_signal:
        print("Mode: R&D Benchmark (Background + Signal)")
        X_bg = load_jsonl(args.input_background)
        X_sig = load_jsonl(args.input_signal)
        if X_bg is None or X_sig is None: return
        y_bg = np.zeros(len(X_bg))
        y_sig = np.ones(len(X_sig))
        X = np.vstack([X_bg, X_sig])
        y = np.concatenate([y_bg, y_sig])
    else:
        print("Error: Missing inputs.")
        return

    if X is None or len(X) == 0: return

    # --- Preprocessing & Training ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training Isolation Forest (n_estimators={args.n_estimators})...")
    clf = IsolationForest(n_estimators=args.n_estimators, contamination=args.contamination, n_jobs=-1, random_state=42)
    clf.fit(X_scaled)

    print("Calculating anomaly scores...")
    scores = -clf.decision_function(X_scaled) # Higher = More Anomalous

    # --- LLM-READABLE REPORTING (Enhanced) ---
    print("\n<tool_result>")
    print(f"Algorithm: Isolation Forest (sklearn)")
    
    if y is not None:
        auc = roc_auc_score(y, scores)
        print(f"Global Performance (ROC AUC): {auc:.4f}")
    
    # --- GLOBAL ANALYSIS (The "Tail" Check) ---
    threshold = np.percentile(scores, 99)
    is_anomaly = scores > threshold

    anomalies = X[is_anomaly]
    normal = X[~is_anomaly]
    
    # Global Peak Finding
    hist, bin_edges = np.histogram(anomalies[:, 0], bins=20)
    peak_mass = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1]) / 2
    

    print(f"\n--- Global Anomalies (Top 1% of entire dataset) ---")
    print(f"Dominant Mass Peak: {peak_mass:.1f} GeV")
    print("Interpretation: In unsupervised mode, global anomalies often correspond to the highest energy tail.")
    
    # Interpretation Guide for the LLM
    print("\n--- Interpretation Guide ---")
    print("NOTE: Isolation Forest detects *rarity*. In collider data, the highest energy events (the tail) are naturally rare.")
    print("WARNING: If the 'Global Mass Peak' above is > 5000 GeV, it is likely just the kinematic tail, NOT a resonance.")
    print("ACTION: Use the 'FOCUSED ANALYSIS' below to check for anomalies specifically inside your target signal region.")

    # --- FOCUSED ANALYSIS (The "Signal" Check) ---
    if args.region_start is not None and args.region_end is not None:
        # Convert TeV -> GeV
        r_start_gev = args.region_start * 1000
        r_end_gev = args.region_end * 1000
        
        print(f"\n--- FOCUSED ANALYSIS (Region: {args.region_start} - {args.region_end} TeV) ---")
        
        # Filter Data to Region (Using Column 0 = m_jj)
        mask_region = (X[:, 0] >= r_start_gev) & (X[:, 0] <= r_end_gev)
        
        if np.sum(mask_region) == 0:
            print("Status: No events found in this region. Cannot analyze.")
        else:
            X_focus = X[mask_region]
            scores_focus = scores[mask_region] # Use the globally trained scores
            
            # Find anomalies LOCALLY within this slice
            # We take the top 1% *of the events in this region*
            thresh_focus = np.percentile(scores_focus, 99)
            is_anom_focus = scores_focus > thresh_focus
            
            anoms_focus = X_focus[is_anom_focus]
            normal_focus = X_focus[~is_anom_focus]
        
            # Local Characterization
            # We require at least 10 anomalous events to avoid characterizing random noise
            if len(anoms_focus) >= 10:
                hist_f, bins_f = np.histogram(anoms_focus[:, 0], bins=10)
                peak_mass_f = (bins_f[np.argmax(hist_f)] + bins_f[np.argmax(hist_f)+1]) / 2
                print(f"Local Mass Peak: Anomalies in this window cluster at {peak_mass_f:.1f} GeV.")
                
                # Substructure
                avg_tau_anom = np.mean(anoms_focus[:, 6])
                avg_tau_norm = np.mean(normal_focus[:, 6]) if len(normal_focus) > 0 else 0
                
                print(f"Local Substructure (Tau21): Anomaly={avg_tau_anom:.3f} vs Normal={avg_tau_norm:.3f}")
                
                # REVISED: Morphology Characterization (No False 'POSITIVE' Verdicts)
                print("\nMorphology Characterization:")
                if avg_tau_anom < avg_tau_norm:
                    print(f" - Substructure: Signal-like (Lower Tau21: {avg_tau_anom:.2f}). Anomalies appear more '2-pronged' than average.")
                else:
                    print(f" - Substructure: Background-like (Higher Tau21: {avg_tau_anom:.2f}). Anomalies appear 'messy' or QCD-like.")

                print("\nCAUTION:")
                print(" - This tool finds the LOCAL statistical outliers (Top 1%).")
                print(" - In pure background, these outliers are often just rare QCD fluctuations.")
                print(" - CONCLUSION: This tool confirms the presence of signal-like candidates, but CANNOT prove significance on its own.")
                print(" - ACTION: Rely on LaCATHODE's significance estimate for the final decision.")

            else:
                print(f"Status: Insufficient statistics ({len(anoms_focus)} anomalies) to characterize pattern.")
    
    else:
        print("\n[Tip]: Provide --region_start and --region_end (in TeV) to get a focused analysis of a specific window.")

    print("</tool_result>")
    
    # --- 4. Visual Plotting (For Humans) ---
    if args.plot:
        # Increased size for better readability and space for the Info Panel
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Score Distribution - ADD REGION-SPECIFIC SCORES
        plt.subplot(2, 2, 1)
        plt.hist(scores, bins=50, density=True, alpha=0.7, color='steelblue', label='All Events')
        
        # Highlight scores from the focused region
        if args.region_start is not None and args.region_end is not None:
            r_start_gev = args.region_start * 1000
            r_end_gev = args.region_end * 1000
            mask_region = (X[:, 0] >= r_start_gev) & (X[:, 0] <= r_end_gev)
            if np.sum(mask_region) > 0:
                scores_in_region = scores[mask_region]
                plt.hist(scores_in_region, bins=50, density=True, alpha=0.7, 
                        color='orange', label=f'Region Events ({args.region_start}-{args.region_end} TeV)')
        
        plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2, label='Anomaly Cut (Top 1%)')
        plt.title("1. Anomaly Score Distribution", fontsize=12, fontweight='bold')
        plt.xlabel("Anomaly Score (Higher is more anomalous)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Mass Profile - ENHANCED WITH SHADED REGION
        plt.subplot(2, 2, 2)
        plt.hist(normal[:, 0], bins=50, density=True, alpha=0.5, color='gray', label='Normal (99%)', range=(0, 6000))
        plt.hist(anomalies[:, 0], bins=50, density=True, alpha=0.6, color='crimson', label='Anomalies (Top 1%)', range=(0, 6000))
        
        # Add shaded region to highlight the search zone
        if args.region_start is not None and args.region_end is not None:
            plt.axvspan(args.region_start * 1000, args.region_end * 1000, 
                       alpha=0.2, color='gold', label='Search Region', zorder=0)
            plt.axvline(args.region_start * 1000, color='orange', linestyle='--', linewidth=2)
            plt.axvline(args.region_end * 1000, color='green', linestyle='--', linewidth=2)
        
        plt.title("2. Invariant Mass ($m_{jj}$) Profile", fontsize=12, fontweight='bold')
        plt.xlabel("Invariant Mass [GeV]")
        plt.ylabel("Density (Normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')

        # Plot 3: Substructure (Tau21) - FILTER TO REGION ONLY
        plt.subplot(2, 2, 3)
        
        if args.region_start is not None and args.region_end is not None:
            # Show substructure ONLY for events in the focused region
            r_start_gev = args.region_start * 1000
            r_end_gev = args.region_end * 1000
            mask_region = (X[:, 0] >= r_start_gev) & (X[:, 0] <= r_end_gev)
            
            if np.sum(mask_region) > 0:
                X_region = X[mask_region]
                scores_region = scores[mask_region]
                thresh_region = np.percentile(scores_region, 99)
                is_anom_region = scores_region > thresh_region
                
                normal_region = X_region[~is_anom_region]
                anoms_region = X_region[is_anom_region]
                
                if len(normal_region) > 0:
                    plt.hist(normal_region[:, 6], bins=30, density=True, alpha=0.5, 
                            color='gray', label='Normal (In Region)', range=(0,1))
                if len(anoms_region) > 0:
                    plt.hist(anoms_region[:, 6], bins=30, density=True, alpha=0.6, 
                            color='crimson', label='Anomalies (In Region)', range=(0,1))
                plt.title(f"3. Jet Substructure (Tau21) - {args.region_start}-{args.region_end} TeV", 
                         fontsize=12, fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No events in region', ha='center', va='center', fontsize=14)
                plt.title("3. Jet Substructure (Tau21) - NO DATA", fontsize=12, fontweight='bold')
        else:
            # Original global view if no region specified
            plt.hist(normal[:, 6], bins=30, density=True, alpha=0.5, color='gray', 
                    label='Normal (Background-like)', range=(0,1))
            plt.hist(anomalies[:, 6], bins=30, density=True, alpha=0.6, color='crimson', 
                    label='Anomalies (Candidates)', range=(0,1))
            plt.title("3. Jet Substructure (Tau21)", fontsize=12, fontweight='bold')
        
        plt.xlabel(r"$\tau_{21}$ (Lower $\leftarrow$ Signal-like | Background-like $\rightarrow$ Higher)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')

        # Plot 4: Info Panel (New)
        # This replaces the empty slot with a guide on how to interpret the results
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        info_text = (
            r"$\bf{INTERPRETATION\ GUIDE}$" + "\n\n"
            r"$\bf{Fig\ 1:\ Score\ Distribution}$" + "\n"
            "Events to the right of the red line are selected as anomalies.\n\n"
            r"$\bf{Fig\ 2:\ Mass\ Bump\ Hunt\ (Crucial)}$" + "\n"
            "Look for a sharp peak (resonance) in the Red histogram.\n"
            "If Red just follows Gray (but shifted right), it's likely\n"
            "just the kinematic tail (not new physics).\n\n"
            r"$\bf{Fig\ 3:\ Substructure\ Check}$" + "\n"
            "Real signals often have 2-prong structure (Low Tau21).\n"
            "- If Red peaks at low values (<0.4): Strong Signal Evidence.\n"
            "- If Red peaks at high values (>0.6): Likely QCD Noise."
        )
        ax4.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top', linespacing=1.4)

        # DEBUG INFO (Bottom Right)
        # Discrete footer for tracking experiments
        input_name = "Unknown"
        if args.input_unlabeled: input_name = os.path.basename(args.input_unlabeled)
        elif args.input_background: input_name = "R&D Benchmark"
        
        region_str = "Full Spectrum"
        if args.region_start: region_str = f"{args.region_start} - {args.region_end} TeV"
        
        debug_text = (
            f"RUN CONFIGURATION:\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Input: {input_name}\n"
            f"Region: {region_str}\n"
            f"Params: Trees={args.n_estimators}, Contam={args.contamination}\n"
            f"Job ID: {args.job_id if args.job_id else 'N/A'}"
        )
        
        # Place debug text in bottom right of the figure canvas
        plt.figtext(0.98, 0.02, debug_text, ha='right', va='bottom', 
                    fontsize=9, color='dimgray', family='monospace', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

        # Final Layout Adjustments
        plt.suptitle("Isolation Forest Anomaly Detection Report", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Leave room for header/footer
        
        # Save
        save_dir = ['toolout', 'graphs', 'iforest']
        session_id = os.environ.get("FRAMEWORK_SESSION_ID", None)
        if session_id:
            save_dir.append(f"session_{session_id}")
        
        if args.job_id:
            save_dir.append(f"job_{args.job_id}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(*save_dir, f"isolation_forest_analysis_{timestamp}.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150) # Higher DPI for clearer text
        print(f"Detailed physics profile plot saved to {save_path}")

if __name__ == "__main__":
    main()