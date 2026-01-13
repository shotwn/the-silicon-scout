from argparse import ArgumentParser
import numpy as np
import json
import os
import framework.tools.lacathode_event_dictionary as lacathode_event_dictionary

from framework.logger import get_logger

"""
Argument parser for running the script from command line
"""

parser = ArgumentParser()
parser.add_argument('--input_background', type=str, required=False,
                    help='Path to background data file (output of import_and_fastjet.py)')
parser.add_argument('--input_signal', type=str, required=False,
                    help='Path to signal data file (output of import_and_fastjet.py)')
parser.add_argument('--input_unlabeled', type=str, required=False,
                    help='Path to unlabeled data file (output of import_and_fastjet.py), for inference mode')

parser.add_argument('--output_dir', type=str, required=False,
                    default='./toolout/lacathode_input_data/',
                    help='Output directory for LaCATHODE prepared data files. Job ID subdirectory will be created inside this path.')
parser.add_argument('--job_id', type=str, required=True,
                    help='Unique job ID for this run, used to create output subdirectory')

parser.add_argument('--run_mode', type=str, choices=['training', 'inference'], default='training',
                    help='Run mode: "training" for preparing training/validation/test sets, "inference" for preparing unlabeled data for inference')

parser.add_argument('--shuffle_seed', type=int, default=42,
                    help='Random seed for shuffling data before splitting')

parser.add_argument('--training_fraction', type=float, default=0.33,
                    help='Fraction of data to use for training')

parser.add_argument('--validation_fraction', type=float, default=0.33,
                    help='Fraction of data to use for validation')

# For defining the signal region (SR) window
parser.add_argument('--side_band_min', type=float, default=2.0,
                    help='Minimum mass for Sideband (SB) region in TeV')

parser.add_argument('--min_mass', type=float, default=3.3,
                    help='Minimum mass for Signal Region (SR) window in TeV')
parser.add_argument('--max_mass', type=float, default=3.7,
                    help='Maximum mass for Signal Region (SR) window in TeV')

parser.add_argument('--side_band_max', type=float, default=5.0,
                    help='Maximum mass for Sideband (SB) region in TeV')

parser.add_argument('--tho_21_threshold', type=float, default=None,
                    help='Threshold for Tau2/1 ratio filtering (example feature)')

args = parser.parse_args()

"""
Input data
Expected input is the output of import_and_fastjet.py
jsonl file with one entry per event, e.g.:
{
    "type": "background",
    "jets": [{
        "px": 1185.0550829874608, 
        "py": 492.23967647028195, 
        "pz": 83.45395978844432, 
        "m": 63.16421456404189, 
        "n_particles": 33, 
        "P_T_lead": 402.4897766113281, 
        "tau_1": 0.028296031700895678, 
        "tau_2": 0.019215605199032395, 
        "tau_3": 0.014383409105619378, 
        "tau_4": 0.011324072483764649
    }, {
        "px": -1195.9301781079894, 
        "py": -474.78326170135693, 
        "pz": 240.06963161424423, 
        "m": 106.91212895959227, 
        "n_particles": 36, 
        "P_T_lead": 847.8291625976561, 
        "tau_1": 0.02403841459865246, 
        "tau_2": 0.0129474846780376, 
        "tau_3": 0.010382179784707351, 
        "tau_4": 0.009467764766421935
    }], 
    "n_particles": 109, 
    "m_jj": 170.07634352363416,
    "dR": 3.141592653589793
    }
"""

class LaCATHODEPreperation:
    def __init__(self, **args):
        self.input_background = args.get('input_background')
        self.input_signal = args.get('input_signal')
        self.input_unlabeled = args.get('input_unlabeled')

        self.job_id = args.get('job_id')
        self.top_output_dir = args.get('output_dir', './toolout/lacathode_input_data/')
        self.output_dir = os.path.join(self.top_output_dir, self.job_id)

        self.shuffle_seed = args.get('shuffle_seed', 42)

        self.training_fraction = args.get('training_fraction', 0.33)
        self.validation_fraction = args.get('validation_fraction', 0.33)
        self.test_fraction = 1.0 - self.training_fraction - self.validation_fraction

        self.run_mode = args.get('run_mode', 'training')  # 'training' or 'inference'

        self.min_mass = args.get('min_mass', 3.3)
        self.max_mass = args.get('max_mass', 3.7)
        self.side_band_min = args.get('side_band_min', 2.0)
        self.side_band_max = args.get('side_band_max', 5.0)

        # Safety Checks for SR and SB definitions
        # Logical Consistency & Units
        if self.min_mass >= self.max_mass:
            raise ValueError(f"Signal Region Error: min_mass ({self.min_mass}) must be smaller than max_mass ({self.max_mass}).")
            
        if self.side_band_min >= self.side_band_max:
             raise ValueError(f"Sideband Error: side_band_min ({self.side_band_min}) must be smaller than side_band_max ({self.side_band_max}).")

        if any(x > 10.0 for x in [self.min_mass, self.max_mass, self.side_band_min, self.side_band_max]):
            raise ValueError("Unit Mismatch: Mass values > 10.0 detected. Please provide inputs in TeV (e.g. 3.5), not GeV.")

        # Containment Checks
        if self.side_band_min >= self.min_mass:
            raise ValueError(f"Left Sideband Missing! side_band_min ({self.side_band_min}) must be strictly lower than SR start ({self.min_mass}).")
            
        if self.side_band_max <= self.max_mass:
            raise ValueError(f"Right Sideband Missing! side_band_max ({self.side_band_max}) must be strictly higher than SR end ({self.max_mass}).")

        # Reliability Checks (Anchors)
        # LaCATHODE needs ~0.5 TeV on BOTH sides to interpolate the background safely.
        left_anchor = self.min_mass - self.side_band_min
        right_anchor = self.side_band_max - self.max_mass
        min_anchor = 0.5 

        if left_anchor < min_anchor:
            raise ValueError(f"Left Sideband too narrow ({left_anchor:.2f} TeV). Needs > {min_anchor} TeV for stable interpolation.")
        
        if right_anchor < min_anchor:
            raise ValueError(f"Right Sideband too narrow ({right_anchor:.2f} TeV). Needs > {min_anchor} TeV to prevent extrapolation instability.")

        # Signal Region Sizing
        sr_width = self.max_mass - self.min_mass
        if sr_width < 0.19:
            raise ValueError(f"Signal Region definition too narrow ({sr_width:.2f} TeV). Window should be at least 0.2 TeV to capture sufficient events.")
        if sr_width > 1.2:
            raise ValueError(f"Signal Region definition too wide ({sr_width:.2f} TeV). Background estimation degrades if window > 1.0 TeV.")

        
        self.tho_21_threshold = args.get('tho_21_threshold', None)  # for Tau2/1 ratio filtering

        self.feature_dictionary = lacathode_event_dictionary.tags

        self.logger = get_logger("LaCATHODEPreperation", level="INFO")
        self.toolout_texts = []

        self.session_id = os.environ.get("FRAMEWORK_SESSION_ID", None)

    def add_toolout_text(self, text):
        self.toolout_texts.append(text)
        self.logger.info(text)


    def load_to_numpy(self, input_file, label_type='background', normalize=True):
        """
        Load data from jsonl file into numpy array
        Sorted by jet mass, lighter jet first
        Each event represented by a row in the numpy array
        20 features per event:
        - For each of the two jets:
            - px, py, pz, m, n_particles, P_T_lead, tau_1, tau_2, tau_3, tau_4
        - Event-level features:
            - n_particles, m_jj, dR
        """

        skipped_events = 0

        try:
            # First count the number of events
            with open(input_file, 'r') as f:
                num_events = sum(1 for _ in f)

            # Create numpy array with the correct shape
            numpy_array = np.empty((num_events, len(lacathode_event_dictionary.tags)))

            event_index = 0
            with open(input_file, 'r') as f:
                # Go line by line to avoid loading everything into memory at once
                # Populate only numpy array in memory
                for line in f:
                    event = json.loads(line)
                    jets = event.get('jets', [])

                    if len(jets) != 2:
                        skipped_events += 1
                        continue

                    # We will sort it so light jet is first
                    jet_a = jets[0]
                    jet_b = jets[1]

                    if jet_a.get('m', 0.0) > jet_b.get('m', 0.0):
                        jets = [jet_b, jet_a]

                    # Calculate difference between leading jet and subleading jet masses
                    mass_diff = jets[1].get('m', 0.0) - jets[0].get('m', 0.0)

                    # Calculate Tau 2/1 ratios for both jets
                    tau2_over_tau1_j1 = jets[0].get('tau_2', 0.0) / (jets[0].get('tau_1', 0.0) + 1e-5) # Avoid division by zero
                    tau2_over_tau1_j2 = jets[1].get('tau_2', 0.0) / (jets[1].get('tau_1', 0.0) + 1e-5) # Avoid division by zero

                    # Apply Tau2/1 filtering if threshold is set
                    if self.tho_21_threshold is not None:
                        if tau2_over_tau1_j1 > self.tho_21_threshold or tau2_over_tau1_j2 > self.tho_21_threshold:
                            # Skip this event
                            continue

                    if normalize:
                        # Normalize features as needed
                        mass_diff /= 1000.0  # Scale mass difference to TeV
                        mj1 = jets[0].get('m', 0.0) / 1000.0  # Scale jet mass to TeV
                        mj2 = jets[1].get('m', 0.0) / 1000.0  # Scale jet mass to TeV
                        mjj = event.get('m_jj', 0.0) / 1000.0  # Scale dijet mass to TeV
                    else:
                        mj1 = jets[0].get('m', 0.0)
                        mj2 = jets[1].get('m', 0.0)
                        mjj = event.get('m_jj', 0.0)

                    # pxj1, pyj1, pzj1, mj1, n_particles_j1, P_T_lead_j1, tau1_j1, tau2_j1, tau3_j1, tau4_j1,
                    # pxj2, pyj2, pzj2, mj2, n_particles_j2, P_T_lead_j2, tau1_j2, tau2_j2, tau3_j2, tau4_j2
                    # n_particles, m_jj, dR
                    # Dimensions: (number of events, 27)
                    # See feature_dictionary for order
                    features = [
                        # Event-level features
                        mjj,
                        event.get('n_particles', 0),
                        event.get('dR', 0.0),
                        mass_diff,
                        # Light jet
                        jets[0].get('px', 0.0),
                        jets[0].get('py', 0.0),
                        jets[0].get('pz', 0.0),
                        mj1,
                        jets[0].get('n_particles', 0),
                        jets[0].get('P_T_lead', 0.0),
                        jets[0].get('tau_1', 0.0),
                        jets[0].get('tau_2', 0.0),
                        jets[0].get('tau_3', 0.0),
                        jets[0].get('tau_4', 0.0),
                        tau2_over_tau1_j1,
                        # Heavy jet
                        jets[1].get('px', 0.0),
                        jets[1].get('py', 0.0),
                        jets[1].get('pz', 0.0),
                        mj2,
                        jets[1].get('n_particles', 0),
                        jets[1].get('P_T_lead', 0.0),
                        jets[1].get('tau_1', 0.0),
                        jets[1].get('tau_2', 0.0),
                        jets[1].get('tau_3', 0.0),
                        jets[1].get('tau_4', 0.0),
                        tau2_over_tau1_j2,

                        # Label: 0 for background, 1 for signal
                        1.0 if label_type == 'signal' else 0.0,
                    ]

                    numpy_array[event_index, :] = features
                    event_index += 1

        except Exception as e:
            self.add_toolout_text(f"Error loading data from {input_file}: {e}")
            return None

        if skipped_events > 0:
            self.add_toolout_text(f"Skipped {skipped_events} events due to missing or invalid jet data.")
        return numpy_array[:event_index] # Return only populated rows
    
    def shuffle(self, data_array):
        """
        Shuffle the data array
        """
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(data_array)
        return data_array
    
    def shuffle_and_split(self, data_array, train_fraction=None, val_fraction=None):
        """
        Shuffle and split the data into training, validation, and test sets
        """
        data_array = self.shuffle(data_array)

        num_events = data_array.shape[0]
        train_end = int(num_events * train_fraction)
        val_end = train_end + int(num_events * val_fraction)

        train_set = data_array[:train_end]
        val_set = data_array[train_end:val_end]
        test_set = data_array[val_end:]

        return train_set, val_set, test_set
    
    def separate_SB_SR(self, data):
        """
        Splits data into Signal Region (Inner) and Sideband (Outer)
        based on m_jj window.
        Assumes m_jj is at index 0.
        """
        mjj_col = 0
        
        # Create mask: True if inside the window (SR/Inner), False if outside (SB/Outer)
        innermask = (data[:, mjj_col] > self.min_mass) & (data[:, mjj_col] < self.max_mass)
        outermask = ~innermask
        
        return data[innermask], data[outermask]
    
    def cut_sideband_extremes(self, data):
        """
        Cuts events with m_jj outside the sideband extremes
        """
        mjj_col = 0
        
        mask = (data[:, mjj_col] > self.side_band_min) & (data[:, mjj_col] < self.side_band_max)
        
        return data[mask]
    
    def save_numpy(self, data_array, output_file):
        """
        Save numpy array to file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        np.save(output_file, data_array)

    def graph_all_features(self, data_array, data_label='data'):
        """
        Graph all features for visualization/debugging
        """
        import matplotlib.pyplot as plt

        num_features = data_array.shape[1]
        columns = 4
        rows = (num_features + columns - 1) // columns
        plt, axs = plt.subplots(rows, columns, figsize=(columns * 8, rows * 4))
        plt.tight_layout(pad=3.0) # Makes the layout less cramped
        for i in range(num_features):
            axs[i // columns][i % columns].hist(data_array[:, i], bins=50, alpha=0.7)
            axs[i // columns][i % columns].set_title(f' {self.feature_dictionary[i]} Distribution ({data_label})')
            axs[i // columns][i % columns].set_xlabel(f'{self.feature_dictionary[i]} Value')
            axs[i // columns][i % columns].set_ylabel('Counts')
            axs[i // columns][i % columns].grid()

        if self.session_id:
            plt.suptitle(f'Feature Distributions for {data_label} - Session {self.session_id}', fontsize=16)
        else:
            plt.suptitle(f'Feature Distributions for {data_label}', fontsize=16)

        save_dir = [top_dir]
        if self.session_id:
            save_dir.append(self.session_id)
        if data_label:
            save_dir.append(data_label)
        save_prefix = f'{self.session_id + "_" if self.session_id else ""}'
        save_to = os.path.join(GRAPHS_DIR, f'{save_prefix}feature_distribution_{data_label}.png')
        plt.savefig(save_to)

    def apply_filtering(self, data):
        """
        Apply any filtering criteria to the data array
        For example, filter based on Tau2/1 ratio if threshold is set
        """
        # Copy data to avoid modifying original
        data = data.copy()

        initial_count = len(data)

        # Filter non-finite events
        data = data[np.all(np.isfinite(data), axis=1)]

        dropped = initial_count - len(data)
        if dropped > 0:
            self.add_toolout_text(f"Dropped {dropped} non-finite events from raw data.")

        # We fetch indices dynamically using dictionary tags
        # Note: Data here is already normalized (Mass in TeV) by load_to_numpy if normalize=True
        # But looking at load_to_numpy, it normalizes m_jj but keeps jet mass (mj1/mj2) in TeV as well.
        # 10 GeV = 0.01 TeV.
        
        # Fetch indices
        idx_j1_mass = lacathode_event_dictionary.tags.index("j1_mass")
        idx_j2_mass = lacathode_event_dictionary.tags.index("j2_mass")
        idx_j1_tau21 = lacathode_event_dictionary.tags.index("j1_tau2_over_tau1")
        idx_j2_tau21 = lacathode_event_dictionary.tags.index("j2_tau2_over_tau1")

        # Create Mask: Keep only events where BOTH jets have Mass > 10 GeV (0.01 TeV)
        # and valid Tau21 (> 0)
        clean_mask = (data[:, idx_j1_mass] > 0.01) & (data[:, idx_j2_mass] > 0.01)
        clean_mask &= (data[:, idx_j1_tau21] > 0.0) & (data[:, idx_j2_tau21] > 0.0)

        # Apply
        pre_clean_count = len(data)
        data = data[clean_mask]
        cleaned_count = pre_clean_count - len(data)
        
        if cleaned_count > 0:
            self.add_toolout_text(f"CLEANING: Removed {cleaned_count} artifact events (Mass < 10 GeV or Invalid Tau21).")

        return data

    def training_mode(self):
        """
        MODE 1: Proving the Model (Training/Validation/Testing)
        Uses labeled Background and Signal files.
        """
        self.add_toolout_text("Running in TRAINING mode...")
        if not self.input_background or not self.input_signal:
            raise ValueError("input_background and input_signal required for training mode.")

        # Load
        bg = self.load_to_numpy(self.input_background, label_type='background')
        sig = self.load_to_numpy(self.input_signal, label_type='signal')
        
        # Combine and Shuffle
        combined = np.vstack((bg, sig))
        self.add_toolout_text(f"Loaded {bg.shape[0]} background events and {sig.shape[0]} signal events, total {combined.shape[0]} events.")

        # Warn if events less than 1000
        if combined.shape[0] < 10000:
            self.add_toolout_text("WARNING: Total number of events less than 10000. Recheck signal region window and scan range.")

        # Filter non-finite events before splitting
        combined = self.apply_filtering(combined)
        self.add_toolout_text(f"{combined.shape[0]} events remain after filtering non-finite and artifact events.")

        # Cut sideband extremes
        combined = self.cut_sideband_extremes(combined)
        self.add_toolout_text(f"{combined.shape[0]} events remain after cutting sideband extremes ({self.side_band_min} to {self.side_band_max} TeV).")

        # Shuffle and Split
        # This handles the random seed, shuffling, and index slicing
        train_set, val_set, test_set = self.shuffle_and_split(
            combined, 
            train_fraction=self.training_fraction, 
            val_fraction=self.validation_fraction
        )

        # Echo event counts
        self.add_toolout_text(f"Data split into {train_set.shape[0]} training events, {val_set.shape[0]} validation events, and {test_set.shape[0]} testing events.")

        # Separate SR/SB (Inner/Outer)
        # We need "Outer" to train the model, and "Inner" to test it.
        tr_in, tr_out = self.separate_SB_SR(train_set)
        val_in, val_out = self.separate_SB_SR(val_set)
        test_in, test_out = self.separate_SB_SR(test_set)

        self.add_toolout_text(f"Training set: {tr_in.shape[0]} inner (SR) events, {tr_out.shape[0]} outer (SB) events.")
        self.add_toolout_text(f"Validation set: {val_in.shape[0]} inner (SR) events, {val_out.shape[0]} outer (SB) events.")
        self.add_toolout_text(f"Testing set: {test_in.shape[0]} inner (SR) events, {test_out.shape[0]} outer (SB) events.")

        # Save standard CATHODE files
        self.save_numpy(tr_out, os.path.join(self.output_dir, 'outerdata_train.npy'))
        self.save_numpy(val_out, os.path.join(self.output_dir, 'outerdata_val.npy'))
        self.save_numpy(test_out, os.path.join(self.output_dir, 'outerdata_test.npy'))

        self.save_numpy(tr_in, os.path.join(self.output_dir, 'innerdata_train.npy'))
        self.save_numpy(val_in, os.path.join(self.output_dir, 'innerdata_val.npy'))

        # This is the file we use to "Prove" the model (calculate ROC/SIC)
        self.save_numpy(test_in, os.path.join(self.output_dir, 'innerdata_test.npy'))

        combined_inner = np.vstack((tr_in, val_in, test_in))
        self.save_numpy(combined_inner, os.path.join(self.output_dir, 'innerdata_combined.npy'))
        combined_outer = np.vstack((tr_out, val_out, test_out))
        self.save_numpy(combined_outer, os.path.join(self.output_dir, 'outerdata_combined.npy'))

        # Graph features for debugging
        self.graph_all_features(tr_out, data_label='outerdata_train')
        self.graph_all_features(val_out, data_label='outerdata_val')
        self.graph_all_features(test_out, data_label='outerdata_test')
        self.graph_all_features(tr_in, data_label='innerdata_train')
        self.graph_all_features(val_in, data_label='innerdata_val')
        self.graph_all_features(test_in, data_label='innerdata_test')
        


        self.add_toolout_text(f"Saved training files: train/val/test splits of innerdata_*.npy and outerdata_*.npy")
        self.add_toolout_text(f"1. Train Flow on: {os.path.join(self.output_dir, 'outerdata_train.npy')}")
        self.add_toolout_text(f"2. Detect anomalies in: {os.path.join(self.output_dir, 'innerdata_train.npy')}")
        self.add_toolout_text(f"3. Validate on: {os.path.join(self.output_dir, 'outerdata_val.npy')} and {os.path.join(self.output_dir, 'innerdata_val.npy')}")
        self.add_toolout_text(f"4. Test on: {os.path.join(self.output_dir, 'outerdata_test.npy')} and {os.path.join(self.output_dir, 'innerdata_test.npy')}")

    def inference_mode(self):
        """
        MODE 2: Actually Using It (True Inference / Black Box)
        Uses one Unlabeled Data file (Real Data).
        Splits it into Train/Val/Test so the Trainer works out-of-the-box.
        """
        self.add_toolout_text("Running in INFERENCE mode...")
        if not self.input_unlabeled:
            raise ValueError("input_unlabeled required for inference mode.")

        # Load unlabeled data (Label defaults to 0 usually, or doesn't matter)
        data = self.load_to_numpy(self.input_unlabeled, label_type='background')
        self.add_toolout_text(f"Loaded {data.shape[0]} unlabeled events from {self.input_unlabeled}.")

        # Filter non-finite events before splitting
        data = self.apply_filtering(data)

        # Cut sideband extremes
        data = self.cut_sideband_extremes(data)
        self.add_toolout_text(f"{data.shape[0]} events remain after cutting sideband extremes ({self.side_band_min} to {self.side_band_max} TeV).")
        
        # Shuffle and Split
        # This handles the random seed, shuffling, and index slicing
        # We won't have test set so give its fraction to training
        train_fraction = 1 - self.validation_fraction
        val_fraction = self.validation_fraction
        
        train_set, val_set, _empty_set = self.shuffle_and_split(
            data, 
            train_fraction=train_fraction, 
            val_fraction=val_fraction
        )

        # Echo event counts
        self.add_toolout_text(f"Unlabeled data split into {train_set.shape[0]} training events and {val_set.shape[0]} validation events.")

        # Separate SR/SB (Inner/Outer)
        # We need "Outer" to train the model, and "Inner" to test it.
        tr_in, tr_out = self.separate_SB_SR(train_set)
        val_in, val_out = self.separate_SB_SR(val_set)

        # Since there are no labels in our data files, we don't need to separate test set
        # We just copy val_set to test_set for code simplicity
        test_in = val_in.copy()
        test_out = val_out.copy()

        self.add_toolout_text(f"Training set: {tr_in.shape[0]} inner (SR) events, {tr_out.shape[0]} outer (SB) events.")
        self.add_toolout_text(f"Validation set: {val_in.shape[0]} inner (SR) events, {val_out.shape[0]} outer (SB) events.")
        self.add_toolout_text(f"Testing set: {test_in.shape[0]} inner (SR) events, {test_out.shape[0]} outer (SB) events.")
        
        # Save
        # 'outerdata_inference_train.npy' -> Use this to train the Flow model on real data sidebands
        self.save_numpy(tr_out, os.path.join(self.output_dir, 'outerdata_train.npy'))
        self.save_numpy(val_out, os.path.join(self.output_dir, 'outerdata_val.npy'))
        
        # 'innerdata_inference_train.npy' -> This is where the anomalies are hidden!
        # The model will generate synthetic background to compare against THIS file.
        self.save_numpy(tr_in, os.path.join(self.output_dir, 'innerdata_train.npy'))
        self.save_numpy(val_in, os.path.join(self.output_dir, 'innerdata_val.npy'))

        # Test set is just a copy of validation set in inference mode
        self.save_numpy(test_in, os.path.join(self.output_dir, 'innerdata_test.npy'))
        self.save_numpy(test_out, os.path.join(self.output_dir, 'outerdata_test.npy'))

        # Save combined file to run Oracle inference on
        combined_inner = np.vstack((tr_in, val_in))
        self.save_numpy(combined_inner, os.path.join(self.output_dir, 'innerdata_combined.npy'))

        combined_outer = np.vstack((tr_out, val_out))
        self.save_numpy(combined_outer, os.path.join(self.output_dir, 'outerdata_combined.npy'))
        
        self.add_toolout_text(f"Saved inference files.")
        self.add_toolout_text(f"1. Train Flow on: {os.path.join(self.output_dir, 'outerdata_train.npy')}")
        self.add_toolout_text(f"2. Detect anomalies in: {os.path.join(self.output_dir, 'innerdata_train.npy')}")
        self.add_toolout_text(f"3. Validate on: {os.path.join(self.output_dir, 'outerdata_val.npy')} and {os.path.join(self.output_dir, 'innerdata_val.npy')}")
        self.add_toolout_text(f"4. Test on: {os.path.join(self.output_dir, 'outerdata_test.npy')} and {os.path.join(self.output_dir, 'innerdata_test.npy')}")

    def run(self):
        if self.run_mode == 'training':
            self.training_mode()
        elif self.run_mode == 'inference':
            self.inference_mode()
        else:
            raise ValueError(f"Unknown run mode: {self.run_mode}")

if __name__ == "__main__":
    # For test run: 
    # py .\framework\tools\lacathode_preperation.py --input_background=fastjet-output/background_events.jsonl --input_signal=fastjet-output/signal_events.jsonl
    preperation = LaCATHODEPreperation(
        **vars(args)
    )
    try:
        preperation.run()
    except Exception as e:
        preperation.add_toolout_text(f"Error during LaCATHODE Preperation: {e}")
        raise e # Signal the worker that an error occurred
    finally:
        # Print toolout texts
        print("<tool_result>")
        print("\n".join(preperation.toolout_texts))
        print("</tool_result>")