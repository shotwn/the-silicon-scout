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

import framework.tools.lacathode_event_dictionary as LEDict
from framework.tools.lacathode_common import LaCATHODEProcessor

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

default_data_dir = os.path.join("toolout", "lacathode_input_data")
default_model_dir = os.path.join("toolout", "lacathode_trained_models")

parser = argparse.ArgumentParser(description="Train LaCATHODE Model")
parser.add_argument("--data_dir", type=str, default=default_data_dir,
                    help="Directory containing processed .npy files")
parser.add_argument("--model_dir", type=str, default=default_model_dir,
                    help="Directory to save/load trained models")
parser.add_argument("--load_flow", action="store_true",
                    help="Load existing Flow model instead of retraining")
parser.add_argument("--load_classifier", action="store_true",
                    help="Load existing Classifier instead of retraining")
parser.add_argument("--epochs_flow", type=int, default=int(os.getenv("LACATHODE_FLOW_EPOCHS", 100)),
                    help="Number of epochs for Flow training")
parser.add_argument("--epochs_clf", type=int, default=int(os.getenv("LACATHODE_CLASSIFIER_EPOCHS", 50)),
                    help="Number of epochs for Classifier training")
parser.add_argument("--plot", action="store_true",
                    help="Generate ROC curve plot after training")
parser.add_argument("--batch_size", type=int, default=int(os.getenv("LACATHODE_BATCH_SIZE", 256)),
                    help="Batch size for training models, adjust based on available memory, 256 is low but safe")
parser.add_argument("--verbose", action="store_true", default=True,
                    help="Enable verbose logging during training")

args = parser.parse_args()

class LaCATHODETrainer:
    def __init__(self, data_dir, model_dir, batch_size=None):
        self.data_dir = data_dir
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
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

        self.processor = LaCATHODEProcessor()

        self.toolout_response = []

        # Determine Device
        ## Either Metal or CUDA if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Batch Size
        # Adjust based on available memory
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = int(os.environ.get("LACATHODE_BATCH_SIZE", 256))

        self.log_toolout(
            (
                f"Using device: {self.device} "
                f"Batch Size: {self.batch_size} "
                f"Flow Max Epochs: {args.epochs_flow} "
                f"Classifier Max Epochs: {args.epochs_clf}\n"
            )
        )
    
    def log_toolout(self, message):
        self.toolout_response.append(message)
        print(message)

    def load_data(self):
        """
        Loads the SR (Inner) and SB (Outer) data created by the preparation script.
        """
        self.log_toolout(f"--- Loading Data from {self.data_dir} ---")
        try:
            # Load Sideband Data (used to train Flow)
            self.outer_train = np.load(os.path.join(self.data_dir, "outerdata_train.npy"))
            self.outer_val = np.load(os.path.join(self.data_dir, "outerdata_val.npy"))
            
            # Load Signal Region Data (used to train Classifier)
            self.inner_train = np.load(os.path.join(self.data_dir, "innerdata_train.npy"))
            self.inner_val = np.load(os.path.join(self.data_dir, "innerdata_val.npy"))
            self.inner_test = np.load(os.path.join(self.data_dir, "innerdata_test.npy"))

            self.log_toolout(f"Loaded Outer Train: {self.outer_train.shape}")
            self.log_toolout(f"Loaded Inner Train: {self.inner_train.shape}")

            # LLM Context: Provide distribution info to confirm data quality
            n_sig = np.sum(self.inner_test[:, -1] == 1)
            n_bkg = np.sum(self.inner_test[:, -1] == 0)
            self.log_toolout(f"Test Set Composition: {len(self.inner_test)} events ({int(n_sig)} Signal, {int(n_bkg)} Background)")
            self.log_toolout(f"Data range check (Mass): min={self.inner_train[:, 0].min():.2f}, max={self.inner_train[:, 0].max():.2f}")
        except FileNotFoundError as e:
            self.log_toolout(f"Error loading data: {e}")
            sys.exit(1)


    def prepare_flow_inputs(self):
        print("--- Preprocessing Flow Inputs ---")
        
        # Extract Raw Features
        x_train_raw = self.outer_train[:, 1:-1]
        x_val_raw   = self.outer_val[:, 1:-1]

        # Raw Mass
        m_train_raw = self.outer_train[:, 0:1]
        m_val_raw   = self.outer_val[:, 0:1]

        # Fit the Processor both x and m
        self.processor.fit_scaler(x_train_raw, m_train_raw)

        # Transform Data (Force float32 for PyTorch)
        self.x_outer_train = self.processor.transform(x_train_raw).astype(np.float32)
        self.x_outer_val   = self.processor.transform(x_val_raw).astype(np.float32)

        # Transform Condition (Scale Mass) <--- NEW FIX
        self.m_outer_train = self.processor.transform_condition(m_train_raw).astype(np.float32)
        self.m_outer_val   = self.processor.transform_condition(m_val_raw).astype(np.float32)
        
        # Sanitize
        self.x_outer_train, self.m_outer_train = self.processor.sanitize(self.x_outer_train, self.m_outer_train)
        self.x_outer_val, self.m_outer_val     = self.processor.sanitize(self.x_outer_val, self.m_outer_val)

        self.log_toolout(f"Processed Train Shape: {self.x_outer_train.shape}")
        if len(self.x_outer_val) == 0:
            self.log_toolout("Error: Validation set is empty after processing! Check logic.")
            raise ValueError("Validation set is empty after processing! Check logic.")

        self.log_toolout("Flow inputs prepared successfully.")

    def train_flow(self, load=False, epochs=100, verbose=False):
        """
        STEP 1: The Generative Model
        Trains a Conditional Normalizing Flow on the Sideband.
        It learns to map complex feature distributions to a simple Gaussian.
        """
        print("--- Initializing Normalizing Flow ---")

        self.log_toolout(f"Flow Settings: Inputs={self.x_outer_train.shape[1]}, Context=Mass")
        self.log_toolout(f"Flow Training Config: Epochs={epochs if not load else '0 (Skipped)'}, " 
                         f"Load State={'True' if load else 'False'}")
        
        num_inputs = self.x_outer_train.shape[1]

        # --- DYNAMIC LR CALCULATION ---
        # This is new but I have to test because large batches can cause instability.
        # Heuristic: Maintain constant gradient energy relative to a safe baseline.
        # Baseline: Batch=256, Epochs=100 -> LR=1e-4
        # Formula: New_LR = Baseline_Energy * (New_Batch / New_Epochs)
        
        # 1. Calculate the target LR
        ref_energy = 1e-4 * (100.0 / 256.0) 
        calculated_lr = ref_energy * (self.batch_size / epochs)
        
        # 2. Safety Clip (prevent explosion on massive batches)
        calculated_lr = min(calculated_lr, 2e-3)
        
        print(f"Auto-Calculated LR: {calculated_lr:.2e} (Batch={self.batch_size}, Epochs={epochs})")
        
        self.flow_model = ConditionalNormalizingFlow(
            save_path=self.model_dir,
            num_inputs=num_inputs,
            early_stopping=True,
            epochs=epochs if not load else 0, # Skip training loop if loading
            verbose=verbose,
            device=self.device,
            # FIX FOR UNSTABLE TRAINING ESPECIALLY ON BLACKBOXES - Anıl
            batch_norm=False,    # <--- CRITICAL: Disable the unstable legacy batch norm
            lr=calculated_lr,
            # FIX ENDS HERE 
            batch_size=self.batch_size
        )
        
        if load:
            self.log_toolout("Loading existing Flow model...")
            self.flow_model.load_best_model()
        else:
            self.log_toolout(f"Training Flow for {epochs} max epochs...")
            # Note: The fit method in sk_cathode handles the training loop
            # We override epochs if provided, otherwise use internal defaults
            self.flow_model.fit(
                self.x_outer_train, self.m_outer_train,
                self.x_outer_val, self.m_outer_val
            )
            self.log_toolout("Flow training complete.")

    def transform_to_latent(self):
        print("--- Transforming Signal Region Data to Latent Space ---")
        
        x_inner_train = self.processor.transform(self.inner_train[:, 1:-1]).astype(np.float32)
        x_inner_val   = self.processor.transform(self.inner_val[:, 1:-1]).astype(np.float32)
        
        # Scale Mass here too!
        m_inner_train = self.processor.transform_condition(self.inner_train[:, 0:1]).astype(np.float32)
        m_inner_val   = self.processor.transform_condition(self.inner_val[:, 0:1]).astype(np.float32)

        self.z_train = self.flow_model.transform(x_inner_train, m=m_inner_train)
        self.z_val = self.flow_model.transform(x_inner_val, m=m_inner_val)
        
        self.z_train = self.processor.sanitize(self.z_train)
        self.z_val = self.processor.sanitize(self.z_val)
        
        self.log_toolout(f"Latent Train Shape (Cleaned): {self.z_train.shape}")

        z_mean = np.mean(self.z_train, axis=0)
        z_std = np.std(self.z_train, axis=0)
        self.log_toolout(f"Latent Space Diagnostics (Goal: Mean~0, Std~1):")
        self.log_toolout(f"  > Actual Mean: {np.mean(z_mean):.4f} (avg across dims)")
        self.log_toolout(f"  > Actual Std:  {np.mean(z_std):.4f} (avg across dims)")

    def prepare_classifier_data(self):
        """
        STEP 3: Sampling & Mixing
        We need two classes for our binary classifier:
        Class 1: Real Data transformed to latent space (contains Signal + Background)
        Class 0: Synthetic Background (sampled directly from Normal Distribution)
        """
        self.log_toolout("--- Preparing Classifier Data ---")
        
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

    def train_classifier(self, load=False, epochs=50, verbose=False):
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
            epochs=epochs if not load else 0,
            verbose=verbose,
            device=self.device,
            batch_size=self.batch_size
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
            self.log_toolout("Classifier training complete.")

    def evaluate(self, plot=False):
        print("--- Evaluating Performance ---")
        self.log_toolout(
            "WARNING: Following evaluation results might not be correct if data received is unlabeled (blackbox) "
            "or different from training data.\n "
            "IMPORTANT: Do your own evaluation via oracle tool."
        )
        
        true_labels = self.inner_test[:, -1]
        
        # Process Test Features
        x_test_raw = self.inner_test[:, 1:-1]
        
        # Use the processor to transform features
        try:
            x_test_scaled = self.processor.transform(x_test_raw)
        except ValueError as e:
            self.log_toolout(f"Preprocessing error: {e}")
            return
        
        m_test = self.processor.transform_condition(self.inner_test[:, 0:1]).astype(np.float32)
        
        # Transform to Latent Space (Flow Model)
        self.log_toolout("Transforming test data to latent space...")
        try:
            # This might generate NaNs/Infs for outliers
            z_test = self.flow_model.transform(x_test_scaled, m=m_test)
        except Exception as e:
            self.log_toolout(f"Warning during flow transformation: {e}")
            return

        # Sanitize Data
        # Check for Infs or NaNs in the latent space and remove them
        is_finite = np.all(np.isfinite(z_test), axis=1)
        n_dropped = np.sum(~is_finite)
        
        if n_dropped > 0:
            self.log_toolout(f"WARNING: Dropping {n_dropped} events due to Infinity/NaN in latent space.")
            z_test = z_test[is_finite]
            true_labels = true_labels[is_finite]
            
            if len(z_test) == 0:
                self.log_toolout("Error: All events were dropped. Cannot evaluate.")
                return

        # Scale Latent Features
        # Now this is safe because we removed the Infs
        z_test_scaled = self.latent_scaler.transform(z_test)
        
        # Predict Anomaly Score
        scores = self.classifier.predict(z_test_scaled)
        
        # Calculate Metric (ROC AUC)
        # Ensure scores are flattened
        scores = scores.flatten()
        
        # Double check for any NaNs in scores themselves
        mask = np.isfinite(scores)
        y_clean = true_labels[mask]
        scores_clean = scores[mask]
        
        if len(y_clean) == 0:
            self.log_toolout("No valid scores to evaluate.")
            return
        
        # Handle Unlabeled/Blackbox Data gracefully
        # If there is no Signal (Label=1) in the test set, we cannot calculate ROC.
        # This prevents the "ROC AUC = nan" confusion.
        if np.sum(y_clean == 1) == 0:
            self.log_toolout("Evaluation Skipped: No signal labels found in test set (Inference/Blackbox Mode).")
            self.log_toolout("This is expected. The model is trained and ready for the Oracle tool.")
            return
        
        fpr, tpr, _ = roc_curve(y_clean, scores_clean)
        roc_auc = auc(fpr, tpr)
        
        self.log_toolout(f"\nResult: ROC AUC = {roc_auc:.4f}")

        # --- DIAGNOSTIC LOGIC ---
        if roc_auc < 0.3:
            self.log_toolout("CRITICAL WARNING: Anti-Learning (AUC < 0.3). Check for 'Poisoned Sideband'.")
            self.log_toolout("The actual anomaly is probably located in your Sideband (training data), not your Signal Region.")
            self.log_toolout("RECOMMENDATION: Shift your Signal Region window LEFT or RIGHT to capture the anomaly inside the 'hole' (SR).")
        
        elif roc_auc > 0.90:
             # THIS IS THE NEW CHECK FOR YOUR SITUATION
             self.log_toolout("CRITICAL WARNING: AUC > 0.90 indicates Flow Model Collapse.")
             self.log_toolout("  > The classifier can perfectly distinguish Real Data from Synthetic Data.")
             self.log_toolout("  > This means the Synthetic Data does not look like physics.")
             self.log_toolout("  > RECOMMENDATION: Retrain Flow Model with lower LR or more epochs.")
        
        elif roc_auc > 0.6:
             self.log_toolout("Interpretation: Good separation. Model is likely working.")
        else:
             self.log_toolout("Interpretation: Low separation. Signal might be too subtle or model needs tuning.")
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sic = tpr / np.sqrt(fpr)
            max_sic = np.nanmax(sic)
        self.log_toolout(f"Result: Max SIC = {max_sic:.4f}")

        if plot:
            plt.figure()
            plt.plot(fpr, tpr, label=f'LaCATHODE (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')

            session_id = os.environ.get("FRAMEWORK_SESSION_ID")
            if session_id:
                save_path = os.path.join('toolout', 'graphs', f'{session_id}_lacathode_roc.png')
                plot_title = f'LaCATHODE ROC Curve ({session_id})'
            else:
                save_path = os.path.join('toolout', 'graphs', 'lacathode_roc.png')
                plot_title = 'LaCATHODE ROC Curve'
            
            plt.title(plot_title)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path)
            self.log_toolout(f"Plot saved to {save_path}")
        
def main():
    trainer = LaCATHODETrainer(args.data_dir, args.model_dir, batch_size=args.batch_size)
    try:
        # Load Data
        trainer.load_data()
        
        # Prepare & Train Flow (Background Model)
        trainer.prepare_flow_inputs()
        trainer.train_flow(load=args.load_flow, epochs=args.epochs_flow, verbose=args.verbose)
        
        # Transform Data to Latent Space
        trainer.transform_to_latent()
        
        # Prepare Classifier Data (Mix Real vs Synthetic)
        trainer.prepare_classifier_data()
        
        # Train Classifier (Anomaly Detector)
        trainer.train_classifier(load=args.load_classifier, epochs=args.epochs_clf, verbose=args.verbose)
        
        # Evaluate
        trainer.evaluate(plot=args.plot)
    except Exception as e:
        trainer.log_toolout(f"An error occurred: {e}")
        raise e
    finally:
        print("<tool_result>")
        print("\n".join(trainer.toolout_response))
        print("</tool_result>")

if __name__ == "__main__":
    main()