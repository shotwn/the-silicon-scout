import argparse
import os
import json
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

# Hyperparameters
parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
parser.add_argument("--contamination", type=str, default="auto", help="Expected anomaly fraction")
parser.add_argument("--plot", action="store_true", help="Generate score distribution plot")

args = parser.parse_args()

def extract_features(event):
    """
    Extracts features from the nested FastJet JSONL format.
    Verified against user sample: 
    {"type": "background", "jets": [{"px": ...}], "m_jj": ..., "dR": ...}
    """
    jets = event.get("jets", [])
    
    # Safety check: We need at least 2 jets for a dijet resonance search
    if len(jets) < 2: 
        return [0.0] * 12 

    j1 = jets[0]
    j2 = jets[1]

    # --- Helper: Convert Cartesian (px, py, pz) to Kinematics (pT, eta) ---
    def get_kinematics(jet_dict):
        px = jet_dict.get("px", 0.0)
        py = jet_dict.get("py", 0.0)
        pz = jet_dict.get("pz", 0.0)
        m  = jet_dict.get("m", 0.0)
        
        # Calculate pT (Transverse Momentum)
        pt = np.sqrt(px**2 + py**2)
        
        # Calculate Eta (Pseudorapidity)
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
        # JSON keys are "tau_1", "tau_2"
        tau1 = jet_dict.get("tau_1", 1e-9) 
        tau2 = jet_dict.get("tau_2", 0.0)
        return tau2 / tau1 if tau1 > 0 else 0.0

    j1_tau21 = get_tau21(j1)
    j2_tau21 = get_tau21(j2)

    # --- Feature Vector Construction ---
    features = [
        event.get("m_jj", 0.0),      # Invariant Mass
        abs(j1_m - j2_m),            # Mass Difference
        event.get("dR", 0.0),        # Delta R
        
        # Jet 1
        j1_pt,
        j1_eta,
        j1_m,
        j1_tau21,
        
        # Jet 2
        j2_pt,
        j2_eta,
        j2_m,
        j2_tau21
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
    
    # --- 1. Data Loading Strategy ---
    if args.input_unlabeled:
        # Inference Mode: Just one blob of data
        print("Mode: Inference (Unlabeled Data)")
        X = load_jsonl(args.input_unlabeled)
        # No labels available
        
    elif args.input_background and args.input_signal:
        # R&D Mode: We construct a mixed dataset to test sensitivity
        print("Mode: R&D Benchmark (Background + Signal)")
        
        X_bg = load_jsonl(args.input_background)
        X_sig = load_jsonl(args.input_signal)
        
        if X_bg is None or X_sig is None:
            print("Failed to load R&D data.")
            return

        # Create Labels: 0 = Background, 1 = Signal
        y_bg = np.zeros(len(X_bg))
        y_sig = np.ones(len(X_sig))
        
        # Combine them
        X = np.vstack([X_bg, X_sig])
        y = np.concatenate([y_bg, y_sig])
        
        print(f"Combined Data: {len(X_bg)} Background + {len(X_sig)} Signal events.")
        
    else:
        print("Error: You must provide either --input_unlabeled OR both --input_background and --input_signal.")
        return

    if X is None or len(X) == 0:
        print("No data loaded. Exiting.")
        return

    # --- 2. Preprocessing ---
    # Isolation Forest cares about distance, so Scaling is CRUCIAL.
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. Training ---
    print(f"Training Isolation Forest (n_estimators={args.n_estimators})...")
    # Note: We train on EVERYTHING (Unsupervised).
    # If the signal is rare (<10%), iForest should isolate it.
    clf = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination, 
        n_jobs=-1, 
        random_state=42
    )
    clf.fit(X_scaled)

    # --- 4. Scoring ---
    print("Calculating anomaly scores...")
    # decision_function: Average path length. 
    # sklearn returns standard convention: negative = anomaly, positive = normal.
    # We FLIP this so High Value = High Anomaly Score (easier for humans/LLM).
    scores = -clf.decision_function(X_scaled)

    # --- 5. Reporting ---
    print("\n<tool_result>")
    print(f"Algorithm: Isolation Forest (sklearn)")
    
    # If we have labels (R&D mode), calculate AUC
    if y is not None:
        auc = roc_auc_score(y, scores)
        print(f"Performance (ROC AUC): {auc:.4f}")
        
        if auc > 0.55:
            print("Status: SUCCESS. Signal events have distinct isolation patterns.")
        elif auc < 0.45:
             print("Status: INCONCLUSIVE. Signal is blending into background (anti-correlated).")
        else:
            print("Status: RANDOM. Cannot distinguish signal from background.")
    else:
        print("Performance: N/A (No labels provided)")

    # General Statistics (Top 1% outliers)
    threshold = np.percentile(scores, 99)
    n_above = np.sum(scores > threshold)
    print(f"Outlier Stats: {n_above} events flagged in top 1% quantile.")
    
    # Suggestion for LLM
    if y is not None:
        # Check mean score difference
        mean_bg = np.mean(scores[y==0])
        mean_sig = np.mean(scores[y==1])
        print(f"Mean Anomaly Score -> Background: {mean_bg:.3f}, Signal: {mean_sig:.3f}")
        if mean_sig > mean_bg:
            print("Observation: Signal events are generally more anomalous than background.")
        else:
            print("Observation: Signal events look 'normal' to this algorithm.")
            
    print("</tool_result>")

    # --- 6. Plotting ---
    if args.plot:
        plt.figure(figsize=(10, 6))
        if y is not None:
            plt.hist(scores[y==0], bins=50, alpha=0.5, label='Background', density=True)
            plt.hist(scores[y==1], bins=50, alpha=0.5, label='Signal', density=True)
        else:
            plt.hist(scores, bins=50, alpha=0.7, label='Unlabeled Data', density=True)
            
        plt.title("Isolation Forest Anomaly Scores")
        plt.xlabel("Anomaly Score (Higher = More Rare)")
        plt.legend()
        
        save_path = f"./toolout/graphs/iforest_scores.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()