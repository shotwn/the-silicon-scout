import os
import sys
import argparse
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import lacathode_event_dictionary as LEDict

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the sk_cathode package
# We go up two levels from framework/tools/ to root/, then into sk_cathode/
sk_cathode_path = os.path.join(current_dir, "../../sk_cathode")

# Add to sys.path if it's not already there
if sk_cathode_path not in sys.path:
    sys.path.append(sk_cathode_path)

# These imports assume you have the sk_cathode library installed
# or the local files available as per the notebook provided.
try:
    from sk_cathode.generative_models.conditional_normalizing_flow_torch import ConditionalNormalizingFlow
    from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier
    from sk_cathode.utils.preprocessing import LogitScaler
except ImportError:
    print("Error: sk_cathode library not found. Please ensure the sk_cathode folder is in your path.")
    sys.exit(1)

"""
LaCATHODE Training Tool
-----------------------
This script performs the two-step training process for LaCATHODE:
1. Train a Conditional Normalizing Flow (CNF) on Sideband (Outer) data to learn the background.
2. Transform Signal Region (Inner) data into Latent Space.
3. Train a Classifier in Latent Space to distinguish Data from pure Gaussian Noise.

Data Structure Assumption (based on data_preparation.py and user query):
- Column 0: Mass (mjj) -> Conditional variable
- Column 1: n_particles (Event-level)
- Column 2: dR (Event-level)
- Column 3: mass_diff (Event-level)
- Columns 4-14: Light jet features
- Columns 15-25: Heavy jet features
- Column -1: Label -> Signal (1) vs Background (0)
"""

parser = argparse.ArgumentParser(description="Train LaCATHODE Model")
parser.add_argument("--data_dir", type=str, default="./lacathode_input_data/",
                    help="Directory containing processed .npy files")
parser.add_argument("--model_dir", type=str, default="./trained_models/",
                    help="Directory to save/load trained models")
parser.add_argument("--load_flow", action="store_true",
                    help="Load existing Flow model instead of retraining")
parser.add_argument("--load_classifier", action="store_true",
                    help="Load existing Classifier instead of retraining")
parser.add_argument("--epochs_flow", type=int, default=100,
                    help="Number of epochs for Flow training")
parser.add_argument("--epochs_clf", type=int, default=50,
                    help="Number of epochs for Classifier training")
parser.add_argument("--plot", action="store_true",
                    help="Generate ROC curve plot after training")

args = parser.parse_args()

class LaCATHODETrainer:
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Preprocessors
        # LogitScaler helps stretch bounded data (like 0-1 scores) to infinity
        # StandardScaler normalizes to mean 0, std 1
        self.outer_scaler = make_pipeline(LogitScaler(), StandardScaler())
        
        # We need a separate scaler for the latent space inputs to the classifier
        self.latent_scaler = StandardScaler()

        # Random seed for reproducibility
        self.random_seed = 42

        # Feature indices to use for the Flow model
        # This is based on domain knowledge to select relevant features
        self.use_indices = [
            LEDict.get_tag_index("j1_mass", -1),           # mj1
            LEDict.get_tag_index("mass_diff", -1),         # delta mjj
            LEDict.get_tag_index("j1_tau2_over_tau1", -1), # tau21,j1
            LEDict.get_tag_index("j2_tau2_over_tau1", -1), # tau21,j2
            LEDict.get_tag_index("dR", -1)                 # delta Rjj
        ]

    def load_data(self):
        """
        Loads the SR (Inner) and SB (Outer) data created by the preparation script.
        """
        print(f"--- Loading Data from {self.data_dir} ---")
        try:
            # Load Sideband Data (used to train Flow)
            self.outer_train = np.load(os.path.join(self.data_dir, "outerdata_train.npy"))
            self.outer_val = np.load(os.path.join(self.data_dir, "outerdata_val.npy"))
            
            # Load Signal Region Data (used to train Classifier)
            self.inner_train = np.load(os.path.join(self.data_dir, "innerdata_train.npy"))
            self.inner_val = np.load(os.path.join(self.data_dir, "innerdata_val.npy"))
            self.inner_test = np.load(os.path.join(self.data_dir, "innerdata_test.npy"))
            
            print(f"Loaded Outer Train: {self.outer_train.shape}")
            print(f"Loaded Inner Train: {self.inner_train.shape}")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def prepare_flow_inputs_ex(self):
        """
        Prepares data for the Flow model.
        The Flow is conditional: it learns p(features | mass).
        """
        print("--- Preprocessing Flow Inputs ---")
        
        # Data Slicing based on mjj being in column 0
        # m = Mass (Conditional)
        # x = Features (Input)

        # Isolate features (exclude mass and label)
        x_train_raw = self.outer_train[:, 1:-1].copy()  # Exclude mass and label
        x_val_raw = self.outer_val[:, 1:-1].copy()  # Exclude mass and label
        
        # Discrete indexes (like n_particles) should be treated carefully
        # They are integers but we treat them as continuous for the Flow
        discrete_indices = [0, 7, 18] # Adjusted for slicing (originally 1, 8, 19)

        np.random.seed(self.random_seed)
        for idx in discrete_indices:
            # Add uniform noise in range [0, 1) to smooth the integers
            # This should help preventing the Flow model collapsing on discrete values
            x_train_raw[:, idx] += np.random.uniform(0, 0.99, size=x_train_raw.shape[0])
            x_train_raw[:, idx] /= 10.0  # Normalize n_particles to [0, 1]
            x_val_raw[:, idx]   += np.random.uniform(0, 0.99, size=x_val_raw.shape[0])
            x_val_raw[:, idx]   /= 10.0  # Normalize n_particles to [0, 1]


        # Now that data is continuous, scaling will work safely
        self.x_outer_train = self.outer_scaler.fit_transform(x_train_raw)
        self.x_outer_val   = self.outer_scaler.transform(x_val_raw)
        
        # Prepare Conditional (Mass is column 0)
        self.m_outer_train = self.outer_train[:, 0:1]
        self.m_outer_val   = self.outer_val[:, 0:1]

        print("Flow inputs prepared (Dequantization + Scaling applied).")


    def prepare_flow_inputs(self):
        """
        Prepares data for the Flow model using Robust, Feature-Specific Preprocessing.
        Replaces generic LogitScaler to prevent crashes on unbounded data.
        """
        print("--- Preprocessing Flow Inputs (Robust Feature-Specific) ---")
        
        # 1. Copy Raw Data
        # [:, 1:-1] excludes Mass (col 0) and Label (last col)
        x_train = self.outer_train[:, 1:-1].copy()
        x_val   = self.outer_val[:, 1:-1].copy()

        # ---------------------------------------------------------
        # GROUP 1: Discrete Variables (N_particles)
        # ACTION: Dequantization (Add noise to make them continuous)
        # Indices: 0 (Event N), 7 (J1 N), 18 (J2 N)
        # ---------------------------------------------------------
        discrete_indices = [
            LEDict.get_tag_index("n_particles", -1),      # Adjusted for slicing
            LEDict.get_tag_index("j1_n_particles", -1),
            LEDict.get_tag_index("j2_n_particles", -1)
        ]

        print(f"Dequantizing discrete features: {discrete_indices}")
        
        np.random.seed(self.random_seed)
        for idx in discrete_indices:
            # Add noise [0, 1) to smooth integers (e.g. 20 -> 20.43)
            x_train[:, idx] += np.random.uniform(0, 1.0, size=x_train.shape[0])
            x_val[:, idx]   += np.random.uniform(0, 1.0, size=x_val.shape[0])

        # ---------------------------------------------------------
        # GROUP 2: Bounded Ratios (Tau21) -> OPTIONAL LOGIT
        # ACTION: Logit Transform (Only if strictly 0 < x < 1)
        # Indices: 13 (J1 Tau21), 24 (J2 Tau21)
        # ---------------------------------------------------------
        ratio_indices = [
            LEDict.get_tag_index("j1_tau2_over_tau1", -1),
            LEDict.get_tag_index("j2_tau2_over_tau1", -1)
        ]
        print(f"Applying Logit transform to ratios: {ratio_indices}")
        
        # Clip to [1e-4, 1 - 1e-4] to avoid infs at 0 or 1
        x_train[:, ratio_indices] = np.clip(x_train[:, ratio_indices], 1e-4, 1 - 1e-4)
        x_val[:, ratio_indices]   = np.clip(x_val[:, ratio_indices],   1e-4, 1 - 1e-4)
        
        # Apply Logit: log(x / (1-x))
        x_train[:, ratio_indices] = np.log(x_train[:, ratio_indices] / (1 - x_train[:, ratio_indices]))
        x_val[:, ratio_indices]   = np.log(x_val[:, ratio_indices]   / (1 - x_val[:, ratio_indices]))

        # ---------------------------------------------------------
        # GROUP 3: Heavy-Tailed Positives (Mass, pT, Raw Taus)
        # ACTION: Log Transform log(x + 1) to squash outliers
        # Includes: j1_mass(5), j1_pT(7), j1_taus(8-11), j2_mass(16), j2_pT(18), j2_taus(19-22)
        # ---------------------------------------------------------
        log_indices = [
            LEDict.get_tag_index("j1_mass", -1),
            LEDict.get_tag_index("j1_P_T_lead", -1),
            LEDict.get_tag_index("j1_tau_1", -1),
            LEDict.get_tag_index("j1_tau_2", -1),
            LEDict.get_tag_index("j1_tau_3", -1),
            LEDict.get_tag_index("j1_tau_4", -1),
            LEDict.get_tag_index("j2_mass", -1),
            LEDict.get_tag_index("j2_P_T_lead", -1),
            LEDict.get_tag_index("j2_tau_1", -1),
            LEDict.get_tag_index("j2_tau_2", -1),
            LEDict.get_tag_index("j2_tau_3", -1),
            LEDict.get_tag_index("j2_tau_4", -1)
        ]

        print(f"Applying Log transform to heavy tails: {log_indices}")
        
        for idx in log_indices:
            # Shift to be positive (some masses might be slightly neg due to errors)
            min_val = min(x_train[:, idx].min(), x_val[:, idx].min())
            offset = max(0, -min_val) + 1.0 # Ensure log(x) > 0
            
            x_train[:, idx] = np.log(x_train[:, idx] + offset)
            x_val[:, idx]   = np.log(x_val[:, idx]   + offset)

        # Use only certain features based on domain knowledge
        # Feature indices based on the assumed structure:
        x_train = x_train[:, self.use_indices]
        x_val   = x_val[:, self.use_indices]

        # 5. Global Standard Scaling
        # We manually use StandardScaler now since we did custom prep
        print("Applying Standard Scaling...")
        self.outer_scaler = StandardScaler()
        self.x_outer_train = self.outer_scaler.fit_transform(x_train)
        self.x_outer_val   = self.outer_scaler.transform(x_val)

        # Fail if any value is -5.0 or 5.0 (indicates something went wrong)
        if np.any(self.x_outer_train <= -5.0) or np.any(self.x_outer_train >= 5.0):
            actual_tag_indices = [idx + 1 for idx in self.use_indices] # +1 to account for mass column excluded earlier
            current_feature_names = [LEDict.tags[i] for i in actual_tag_indices]

            # Find which features caused the problem
            problematic_col_indices = np.unique(np.where(
                (self.x_outer_train <= -5.0) | (self.x_outer_train >= 5.0)
            )[1])

            for col_idx in problematic_col_indices:
                feat_name = current_feature_names[col_idx]
                
                # Grab the actual values that failed for this specific column
                col_data = self.x_outer_train[:, col_idx]
                extreme_mask = (col_data <= -5.0) | (col_data >= 5.0)
                extremes = col_data[extreme_mask]
                
                print(f"-> Feature: '{feat_name}' (Col {col_idx})")
                print(f"   - Max Sigma: {extremes.max():.2f}")
                print(f"   - Min Sigma: {extremes.min():.2f}")
                print(f"   - Outlier Count: {len(extremes)} / {len(col_data)}")
                    
                
            #raise ValueError("Training data has extreme values after scaling. Check preprocessing.")

        """
        # 6. Outlier Clipping (Safety Net)
        # Prevent any remaining crazy values (e.g. 48 sigma) from killing the Flow
        print("Clipping final values to [-5, 5] sigma...")
        self.x_outer_train = np.clip(self.x_outer_train, -5.0, 5.0)
        self.x_outer_val   = np.clip(self.x_outer_val,   -5.0, 5.0)
        """

        # 7. Prepare Conditional (Mass)
        self.m_outer_train = self.outer_train[:, 0:1]
        self.m_outer_val   = self.outer_val[:, 0:1]

        print("Flow inputs prepared successfully.")

    def train_flow(self, load=False, epochs=100):
        """
        STEP 1: The Generative Model
        Trains a Conditional Normalizing Flow on the Sideband.
        It learns to map complex feature distributions to a simple Gaussian.
        """
        print("--- Initializing Normalizing Flow ---")
        num_inputs = self.x_outer_train.shape[1]
        
        self.flow_model = ConditionalNormalizingFlow(
            save_path=self.model_dir,
            num_inputs=num_inputs,
            early_stopping=True,
            epochs=None if not load else 0, # Skip training loop if loading
            verbose=True
        )
        
        if load:
            print("Loading existing Flow model...")
            self.flow_model.load_best_model()
        else:
            print(f"Training Flow for {epochs} max epochs...")
            # Note: The fit method in sk_cathode handles the training loop
            # We override epochs if provided, otherwise use internal defaults
            self.flow_model.fit(
                self.x_outer_train, self.m_outer_train,
                self.x_outer_val, self.m_outer_val
            )
            print("Flow training complete.")

    def transform_to_latent(self):
        """
        STEP 2: The Latent Transformation
        This is the magic of LaCATHODE. We take the Signal Region (Inner) data,
        and pass it through the Flow we just trained on the Sideband.
        
        If the background physics is consistent, the Background events in SR
        should map to a Standard Normal distribution (Gaussian noise) in latent space.
        Signals, however, will map to something else (anomalies).
        """
        print("--- Transforming Signal Region Data to Latent Space ---")
        
        # Scale inner data using the scaler fitted on outer data
        # MODIFICATION: Slice using self.use_indices before transform
        x_inner_train_raw = self.inner_train[:, 1:-1]
        x_inner_train = self.outer_scaler.transform(x_inner_train_raw[:, self.use_indices])
        m_inner_train = self.inner_train[:, 0:1]
        
        x_inner_val_raw = self.inner_val[:, 1:-1]
        x_inner_val = self.outer_scaler.transform(x_inner_val_raw[:, self.use_indices])
        m_inner_val = self.inner_val[:, 0:1]

        # Transform to Latent Space (z)
        # z = Flow(x, conditional=m)
        self.z_train = self.flow_model.transform(x_inner_train, m=m_inner_train)
        self.z_val = self.flow_model.transform(x_inner_val, m=m_inner_val)
        
        print(f"Latent Train Shape: {self.z_train.shape}")

    def prepare_classifier_data(self):
        """
        STEP 3: Sampling & Mixing
        We need two classes for our binary classifier:
        Class 1: Real Data transformed to latent space (contains Signal + Background)
        Class 0: Synthetic Background (sampled directly from Normal Distribution)
        """
        print("--- Preparing Classifier Data ---")
        
        # Generate Synthetic Background (Class 0)
        # Since Flow targets Normal(0,1), we just sample from numpy's randn
        # We oversample background to help training (4x amount of data)
        n_samples = 4 * len(self.z_train)
        z_samples = np.random.randn(n_samples, self.z_train.shape[1])
        
        # Create labels
        # Data = 1
        data_labeled = np.hstack([self.z_train, np.ones((len(self.z_train), 1))])
        # Synthetic = 0
        synth_labeled = np.hstack([z_samples, np.zeros((len(z_samples), 1))])
        
        # Combine and Shuffle
        train_set = np.vstack([data_labeled, synth_labeled])
        train_set = shuffle(train_set, random_state=42)
        
        # Prepare inputs (X) and targets (y)
        # Fit scaler on the mixed training data
        self.X_clf_train = self.latent_scaler.fit_transform(train_set[:, :-1])
        self.y_clf_train = train_set[:, -1]
        
        # Do similar for validation (using simple 1-to-1 mix for val)
        z_samples_val = np.random.randn(len(self.z_val), self.z_val.shape[1])
        val_data_labeled = np.hstack([self.z_val, np.ones((len(self.z_val), 1))])
        val_synth_labeled = np.hstack([z_samples_val, np.zeros((len(z_samples_val), 1))])
        
        val_set = np.vstack([val_data_labeled, val_synth_labeled])
        val_set = shuffle(val_set, random_state=42)
        
        self.X_clf_val = self.latent_scaler.transform(val_set[:, :-1])
        self.y_clf_val = val_set[:, -1]

    def train_classifier(self, load=False, epochs=50):
        """
        STEP 4: Anomaly Detection
        Train a Neural Network to distinguish Real Latent Data from Gaussian Noise.
        High score = Probability of being Real Data (and not Gaussian noise),
        which implies it is likely Signal.
        """
        print("--- Initializing Classifier ---")
        self.classifier = NeuralNetworkClassifier(
            save_path=os.path.join(self.model_dir, "classifier"),
            n_inputs=self.X_clf_train.shape[1],
            early_stopping=True,
            epochs=None if not load else 0,
            verbose=True
        )
        
        if load:
            print("Loading existing Classifier...")
            self.classifier.load_best_model()
        else:
            print(f"Training Classifier for {epochs} max epochs...")
            self.classifier.fit(
                self.X_clf_train, self.y_clf_train,
                self.X_clf_val, self.y_clf_val
            )
            print("Classifier training complete.")

    def evaluate(self, plot=False):
        """
        Evaluate the model on the held-out Test Set.
        """
        print("--- Evaluating Performance ---")
        
        # 1. Prepare Test Data
        # We assume the last column of innerdata_test is the True Label (Signal vs Bkg)
        true_labels = self.inner_test[:, -1]
        
        # 2. Transform Test Features to Latent Space 
        # MODIFICATION: Slice using self.use_indices before transform
        x_test_raw = self.inner_test[:, 1:-1]
        x_test_scaled = self.outer_scaler.transform(x_test_raw[:, self.use_indices])
        m_test = self.inner_test[:, 0:1]
        
        # The flow transform might generate NaNs for crazy outliers, handle carefully
        try:
            z_test = self.flow_model.transform(x_test_scaled, m=m_test)
        except Exception as e:
            print(f"Warning during transformation: {e}")
            return

        # 3. Scale Latent Features
        z_test_scaled = self.latent_scaler.transform(z_test)
        
        # 4. Predict Anomaly Score
        # The output is probability of being "Data" (Signal) vs "Noise" (Background)
        scores = self.classifier.predict(z_test_scaled)
        
        # 5. Calculate Metric (ROC AUC)
        # Remove NaNs if any
        mask = ~np.isnan(scores).flatten()
        y_clean = true_labels[mask]
        scores_clean = scores[mask].flatten()
        
        fpr, tpr, _ = roc_curve(y_clean, scores_clean)
        roc_auc = auc(fpr, tpr)
        
        print(f"\nResult: ROC AUC = {roc_auc:.4f}")
        
        # Calculate Significance Improvement (SIC)
        # SIC = TPR / sqrt(FPR)
        with np.errstate(divide='ignore', invalid='ignore'):
            sic = tpr / np.sqrt(fpr)
            max_sic = np.nanmax(sic)
        print(f"Result: Max SIC = {max_sic:.4f}")

        if plot:
            plt.figure()
            plt.plot(fpr, tpr, label=f'LaCATHODE (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Anomaly Detection Performance')
            plt.legend(loc='lower right')
            plt.savefig('lacathode_roc.png')
            print("Plot saved to lacathode_roc.png")
        
    def predict(self, inference_file_path, save_name):
        """
        Inference Mode:
        1. Loads blackbox data.
        2. Applies the EXACT same transformations (Logit, Log, Scaling) as training.
        3. Transforms to Latent Space (Flow).
        4. Predicts Anomaly Score (Classifier).
        """
        print(f"\n--- Running Inference on {inference_file_path} ---")
        
        # 1. Load Data
        try:
            # Assumes structure: [Mass, Features..., Label] or just [Mass, Features...]
            data = np.load(inference_file_path)
            print(f"Loaded Inference Data: {data.shape}")
        except Exception as e:
            print(f"Error loading inference file: {e}")
            return

        # 2. Preprocess
        # Slice features (assuming same columns as training)
        # Note: If your inference file has no label column at the end, adjust slicing!
        # Here we assume it matches innerdata structure: [Mass, ..., Label]
        m_raw = data[:, 0:1]
        x_raw = data[:, 1:-1].copy() # Exclude mass (0) and label (-1)

        # A. Discrete Dequantization (Add same noise logic)
        discrete_indices = [
            LEDict.get_tag_index("n_particles", -1),
            LEDict.get_tag_index("j1_n_particles", -1),
            LEDict.get_tag_index("j2_n_particles", -1)
        ]
        # For inference, we usually add 0.5 (mean) or random noise. 
        # Using random noise matches the training distribution best for Flows.
        np.random.seed(42) 
        for idx in discrete_indices:
            x_raw[:, idx] += np.random.uniform(0, 1.0, size=x_raw.shape[0])

        # B. Logit Transform
        ratio_indices = [
            LEDict.get_tag_index("j1_tau2_over_tau1", -1),
            LEDict.get_tag_index("j2_tau2_over_tau1", -1)
        ]
        x_raw[:, ratio_indices] = np.clip(x_raw[:, ratio_indices], 1e-4, 1 - 1e-4)
        x_raw[:, ratio_indices] = np.log(x_raw[:, ratio_indices] / (1 - x_raw[:, ratio_indices]))

        # C. Log Transform (Using SAVED offsets)
        if not hasattr(self, 'log_offsets'):
            print("Error: Model not initialized. You must run prepare_flow_inputs first.")
            return

        for idx, offset in self.log_offsets.items():
            x_raw[:, idx] = np.log(x_raw[:, idx] + offset)

        # D. Standard Scaling (Using PRE-FITTED scaler)
        # Filter only the features we use
        x_selected = x_raw[:, self.use_indices]
        x_scaled = self.outer_scaler.transform(x_selected)

        # 3. Transform to Latent Space
        print("Transforming to Latent Space...")
        z_inference = self.flow_model.transform(x_scaled, m=m_raw)

        # 4. Predict
        # Scale latent inputs using the PRE-FITTED latent scaler
        z_scaled = self.latent_scaler.transform(z_inference)
        
        print("Calculating Anomaly Scores...")
        scores = self.classifier.predict(z_scaled)

        # 5. Save Results
        output_path = os.path.join(self.model_dir, save_name)
        np.save(output_path, scores)
        print(f"Done! Scores saved to: {output_path}")
        print(f"Mean Score: {np.nanmean(scores):.4f} (Higher = More Anomalous)")

def main():
    trainer = LaCATHODETrainer(args.data_dir, args.model_dir)
    
    # 1. Load Data
    trainer.load_data()
    
    # 2. Prepare & Train Flow (Background Model)
    trainer.prepare_flow_inputs()
    trainer.train_flow(load=args.load_flow, epochs=args.epochs_flow)
    
    # 3. Transform Data to Latent Space
    trainer.transform_to_latent()
    
    # 4. Prepare Classifier Data (Mix Real vs Synthetic)
    trainer.prepare_classifier_data()
    
    # 5. Train Classifier (Anomaly Detector)
    trainer.train_classifier(load=args.load_classifier, epochs=args.epochs_clf)
    
    # 6. Evaluate
    trainer.evaluate(plot=args.plot)

if __name__ == "__main__":
    main()