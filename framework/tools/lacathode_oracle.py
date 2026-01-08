import argparse
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from lacathode_common import LaCATHODEProcessor # <--- Uses the common file

# --- Path Setup for sk_cathode ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sk_cathode_path = os.path.join(current_dir, "../../sk_cathode")
if sk_cathode_path not in sys.path:
    sys.path.append(sk_cathode_path)

from sk_cathode.generative_models.conditional_normalizing_flow_torch import ConditionalNormalizingFlow
from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier

class LaCATHODEOracle:
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Instantiate the shared processor
        self.processor = LaCATHODEProcessor()

        # Instantiate the latent scaler (For Outputs: Z)
        self.latent_scaler = StandardScaler()

        print("--- Initializing Oracle State ---")
        self.tool_out = []
        self._restore_state()
    
    def add_tool_out(self, result):
        self.tool_out.append(result)
        print(result)

    def _restore_state(self):
        """
        To predict correctly, we must 'remember' the scaling from the training phase.
        We do this by loading the original training data and re-fitting the processor.
        """
        # Load Original Training Data
        print("1. Loading Training Data (to restore scaler state)...")
        try:
            # We need the Sideband (Outer) to fit the Feature Scaler
            # If there is inference mode, we use the inference training file
            outer_train_path = os.path.join(self.data_dir, "outerdata_inference_train.npy")
            if os.path.exists(outer_train_path):
                outer_train = np.load(outer_train_path)
            else:
                outer_train = np.load(os.path.join(self.data_dir, "outerdata_train.npy"))
            
            # We need the Signal Region (Inner) to fit the Latent Scaler
            # If inference mode, use inference training file
            inner_train_path = os.path.join(self.data_dir, "innerdata_inference_train.npy")
            if os.path.exists(inner_train_path):
                inner_train = np.load(inner_train_path)
            else:
                inner_train = np.load(os.path.join(self.data_dir, "innerdata_train.npy"))
        except FileNotFoundError:
            print("Error: Training data not found. The Oracle needs 'outerdata_train.npy' to calibrate itself.")
            sys.exit(1)

        # Fit the Processor (Learn Log-Offsets and Mean/Std from Training Data)
        # Pass BOTH Features (1:-1) and Mass (0:1) so the condition scaler is fitted
        self.processor.fit_scaler(outer_train[:, 1:-1], outer_train[:, 0:1])

        # Load the Trained Flow Model
        print("2. Loading Flow Model...")
        self.flow_model = ConditionalNormalizingFlow(
            save_path=self.model_dir, 
            num_inputs=len(self.processor.use_indices), 
            epochs=0, 
            verbose=False,
            batch_norm=False,  # Match training config
        )
        self.flow_model.load_best_model()

        # Fit Latent Scaler
        # The Classifier expects inputs normalized to Mean=0, Std=1. 
        # We must transform the training data to latent space to learn these stats.
        print("3. Calibrating Latent Space...")
        x_inner_scaled = self.processor.transform(inner_train[:, 1:-1]).astype(np.float32)
        # Scale the Mass (Condition)
        m_inner = self.processor.transform_condition(inner_train[:, 0:1]).astype(np.float32)
        
        # Project training data to latent space
        z_train = self.flow_model.transform(x_inner_scaled, m=m_inner)
        
        # Sanitize (Remove rare NaNs if any)
        z_train = self.processor.sanitize(z_train)
        
        # The classifier was trained on a mix of Data + Synthetic Gaussian. 
        # We replicate that mix here to get the exact same scaler.
        z_mix = np.vstack([z_train, np.random.randn(*z_train.shape)])
        self.latent_scaler.fit(z_mix)
        
        # Load Classifier
        print("4. Loading Classifier...")
        self.classifier = NeuralNetworkClassifier(
            save_path=os.path.join(self.model_dir, "classifier"),
            n_inputs=z_train.shape[1], 
            epochs=0, 
            verbose=False,
        )
        self.classifier.load_best_model()
        print("Oracle Ready.\n")

    def predict(self, inference_file_path, scores_file_path):
        print(f"--- Running Inference on {inference_file_path} ---")
        
        try:
            data = np.load(inference_file_path)
            self.add_tool_out(f"Loaded {len(data)} events.")
        except Exception as e:
            self.add_tool_out(f"Error loading file: {e}")
            return

        # Prepare Inputs
        # Assumption: Inference file has [Mass, Features..., (Label?)]
        # If it's a blackbox, it might not have a label.
        # We assume standard format: Col 0 is Mass.
        m_raw = data[:, 0:1]
        
        # Check dimensions to decide slicing
        # If columns == 27 (with label), slice 1:-1
        # If columns == 26 (no label), slice 1:
        if data.shape[1] == 27:
            x_raw = data[:, 1:-1]
        else:
            x_raw = data[:, 1:] # Assume all remaining are features

        # Preprocess (Using the restored processor)
        x_scaled = self.processor.transform(x_raw).astype(np.float32)
        # Scale the Mass (Condition)
        m_scaled = self.processor.transform_condition(m_raw).astype(np.float32)

        # Flow Transform (Data -> Latent Z)
        print("Transforming to Latent Space...")
        z_inference = self.flow_model.transform(x_scaled, m=m_scaled)

        # --- SAFETY FIX STARTS HERE ---
        # Check for NaNs/Infs in Latent Space
        # This prevents the ValueError from crashing the script
        valid_mask = np.all(np.isfinite(z_inference), axis=1)
        n_invalid = len(z_inference) - np.sum(valid_mask)
        
        # Initialize output array with a default safe score (0.0 = Background)
        final_scores = np.zeros(len(data), dtype=np.float32)

        if n_invalid > 0:
            self.add_tool_out(f"WARNING: {n_invalid} events produced Infinity/NaN in latent space.")
            self.add_tool_out("         Assigning default score 0.0 to these events.")

        # Classifier (Latent Z -> Anomaly Score)
        # Only process valid events to avoid crash
        if np.any(valid_mask):
            z_valid = z_inference[valid_mask]
            
            # Scale Z using the restored latent scaler
            z_scaled_valid = self.latent_scaler.transform(z_valid)
            
            # Predict
            scores_valid = self.classifier.predict(z_scaled_valid)
            
            # Fill the valid spots in the final array
            final_scores[valid_mask] = scores_valid.flatten()
        
        # --- SAFETY FIX ENDS HERE ---

        # Save
        output_path = os.path.join(scores_file_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.save(output_path, final_scores)
        
        self.add_tool_out(f"Successfully done. Scores saved to: {output_path}")
        self.add_tool_out(f"Mean Score: {np.nanmean(final_scores):.4f} (Higher > 0.5 = More Anomalous)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaCATHODE Oracle Inference")
    # ... [Keep existing arguments] ...
    parser.add_argument("--data_dir", type=str, default="toolout/lacathode_input_data/",
                        help="Directory containing ORIGINAL training data (needed for calibration)")
    parser.add_argument("--model_dir", type=str, default="toolout/lacathode_trained_models/",
                        help="Directory containing trained models")
    parser.add_argument("--inference_file", type=str, required=True,
                        help="Path to the file to predict on (e.g., innerdata_inference.npy)")
    parser.add_argument("--output_file", type=str, default="oracle_scores.npy",
                        help="Name of the output score file")
    
    args = parser.parse_args()

    oracle = None
    try:
        oracle = LaCATHODEOracle(args.data_dir, args.model_dir)
        oracle.predict(args.inference_file, args.output_file)
    except Exception as e:
        # CRITICAL FIX: Add the error to the tool output list 
        # so it gets wrapped in <tool_result> tags.
        error_message = f"CRITICAL ERROR during Oracle inference: {str(e)}"
        if oracle:
            oracle.add_tool_out(error_message)
        else:
            # If init failed, oracle is None, so we print manually
            print(error_message)
            print(f"<tool_result>\n{error_message}\n</tool_result>")
            sys.exit(1) # Signal failure to worker
            
    finally:
        # Safety check: Only print tags if oracle exists
        if oracle:
            print("<tool_result>")
            print('\n'.join(oracle.tool_out))
            print("</tool_result>")