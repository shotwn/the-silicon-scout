from argparse import ArgumentParser
import numpy as np
import json
import os
import lacathode_event_dictionary

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
                    help='Output directory for LaCATHODE prepared data files')

parser.add_argument('--run_mode', type=str, choices=['training', 'inference'], default='training',
                    help='Run mode: "training" for preparing training/validation/test sets, "inference" for preparing unlabeled data for inference')

parser.add_argument('--shuffle_seed', type=int, default=42,
                    help='Random seed for shuffling data before splitting')

parser.add_argument('--training_fraction', type=float, default=0.33,
                    help='Fraction of data to use for training')

parser.add_argument('--validation_fraction', type=float, default=0.33,
                    help='Fraction of data to use for validation')

# For defining the signal region (SR) window
parser.add_argument('--side_band_min', type=float, default=2.5,
                    help='Minimum mass for Sideband (SB) region in TeV')

parser.add_argument('--min_mass', type=float, default=3.3,
                    help='Minimum mass for Signal Region (SR) window in TeV')
parser.add_argument('--max_mass', type=float, default=3.8,
                    help='Maximum mass for Signal Region (SR) window in TeV')

parser.add_argument('--side_band_max', type=float, default=4.0,
                    help='Maximum mass for Sideband (SB) region in TeV')

args = parser.parse_args()

GRAPHS_DIR = 'toolout/graphs/'
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR, exist_ok=True)

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

        self.output_dir = args.get('output_dir', './toolout/lacathode_input_data/')

        self.shuffle_seed = args.get('shuffle_seed', 42)

        self.training_fraction = args.get('training_fraction', 0.33)
        self.validation_fraction = args.get('validation_fraction', 0.33)
        self.test_fraction = 1.0 - self.training_fraction - self.validation_fraction

        self.run_mode = args.get('run_mode', 'training')  # 'training' or 'inference'

        self.min_mass = args.get('min_mass', 3.3)
        self.max_mass = args.get('max_mass', 3.8)
        self.side_band_min = args.get('side_band_min', 2.5)
        self.side_band_max = args.get('side_band_max', 4.0)

        if self.side_band_min >= self.min_mass or self.side_band_max <= self.max_mass:
            print(f"{self.side_band_min} {self.min_mass} {self.max_mass} {self.side_band_max}")
            raise ValueError("Sideband extremes must be outside the Signal Region window.")

        self.feature_dictionary = lacathode_event_dictionary.tags

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
                        print(f"Skipping event {event_index} in {input_file} due to unexpected number of jets: {len(jets)}")
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
            print(f"Error loading data from {input_file}: {e}")
            return None

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

        if not os.path.exists('graphs'):
            os.makedirs('graphs')

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

        plt.savefig(f'toolout/graphs/feature_distribution_{data_label}.png')

    def training_mode(self):
        """
        MODE 1: Proving the Model (Training/Validation/Testing)
        Uses labeled Background and Signal files.
        """
        print("Running in TRAINING mode (Proving the model)...")
        if not self.input_background or not self.input_signal:
            raise ValueError("input_background and input_signal required for training mode.")

        # Load
        bg = self.load_to_numpy(self.input_background, label_type='background')
        sig = self.load_to_numpy(self.input_signal, label_type='signal')
        
        # Combine and Shuffle
        combined = np.vstack((bg, sig))

        # Filter non-finite events before splitting
        initial_count = len(combined)
        combined = combined[np.all(np.isfinite(combined), axis=1)]
        dropped = initial_count - len(combined)
        if dropped > 0:
            print(f"Dropped {dropped} non-finite events from raw data.")

        # Cut sideband extremes
        combined = self.cut_sideband_extremes(combined)

        # Shuffle and Split
        # This handles the random seed, shuffling, and index slicing
        train_set, val_set, test_set = self.shuffle_and_split(
            combined, 
            train_fraction=self.training_fraction, 
            val_fraction=self.validation_fraction
        )

        # Separate SR/SB (Inner/Outer)
        # We need "Outer" to train the model, and "Inner" to test it.
        tr_in, tr_out = self.separate_SB_SR(train_set)
        val_in, val_out = self.separate_SB_SR(val_set)
        test_in, test_out = self.separate_SB_SR(test_set)

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
        


        print("<tool_result>")
        print(f"Saved training files: train/val/test splits of innerdata_*.npy and outerdata_*.npy")
        print(f"1. Train Flow on: {os.path.join(self.output_dir, 'outerdata_train.npy')}")
        print(f"2. Detect anomalies in: {os.path.join(self.output_dir, 'innerdata_train.npy')}")
        print(f"3. Validate on: {os.path.join(self.output_dir, 'outerdata_val.npy')} and {os.path.join(self.output_dir, 'innerdata_val.npy')}")
        print(f"4. Test on: {os.path.join(self.output_dir, 'outerdata_test.npy')} and {os.path.join(self.output_dir, 'innerdata_test.npy')}")
        print("</tool_result>")

    def inference_mode(self):
        """
        MODE 2: Actually Using It (True Inference / Black Box)
        Uses one Unlabeled Data file (Real Data).
        Splits it into Train/Val/Test so the Trainer works out-of-the-box.
        """
        print("Running in INFERENCE mode (Using on real data)...")
        if not self.input_unlabeled:
            raise ValueError("input_unlabeled required for inference mode.")

        # Load unlabeled data (Label defaults to 0 usually, or doesn't matter)
        data = self.load_to_numpy(self.input_unlabeled, label_type='background')

        # Filter non-finite events before splitting
        initial_count = len(data)
        data = data[np.all(np.isfinite(data), axis=1)]
        dropped = initial_count - len(data)
        if dropped > 0:
            print(f"Dropped {dropped} non-finite events from raw data.")

        # Cut sideband extremes
        data = self.cut_sideband_extremes(data)
        
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

        # Separate SR/SB (Inner/Outer)
        # We need "Outer" to train the model, and "Inner" to test it.
        tr_in, tr_out = self.separate_SB_SR(train_set)
        val_in, val_out = self.separate_SB_SR(val_set)
        
        # Since there are no labels in our data files, we don't need to separate test set
        # We just copy val_set to test_set for code simplicity
        test_in = val_in.copy()
        test_out = val_out.copy()
        
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
        
        print("<tool_result>")
        print(f"Saved inference files.")
        print(f"1. Train Flow on: {os.path.join(self.output_dir, 'outerdata_train.npy')}")
        print(f"2. Detect anomalies in: {os.path.join(self.output_dir, 'innerdata_train.npy')}")
        print(f"3. Validate on: {os.path.join(self.output_dir, 'outerdata_val.npy')} and {os.path.join(self.output_dir, 'innerdata_val.npy')}")
        print(f"4. Test on: {os.path.join(self.output_dir, 'outerdata_test.npy')} and {os.path.join(self.output_dir, 'innerdata_test.npy')}")
        print("</tool_result>")

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
    preperation.run()