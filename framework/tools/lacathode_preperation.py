from argparse import ArgumentParser
import numpy as np
import json
import os

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
                    default='./lacathode_input_data/',
                    help='Output directory for LaCATHODE prepared data files')

parser.add_argument('--run_mode', type=str, choices=['training', 'inference'], default='training',
                    help='Run mode: "training" for preparing training/validation/test sets, "inference" for preparing unlabeled data for inference')

parser.add_argument('--shuffle_seed', type=int, default=42,
                    help='Random seed for shuffling data before splitting')

parser.add_argument('--training_fraction', type=float, default=0.33,
                    help='Fraction of data to use for training')

parser.add_argument('--validation_fraction', type=float, default=0.33,
                    help='Fraction of data to use for validation')

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

        self.output_dir = args.get('output_dir', './lacathode_input_data/')

        self.shuffle_seed = args.get('shuffle_seed', 42)

        self.training_fraction = args.get('training_fraction', 0.33)
        self.validation_fraction = args.get('validation_fraction', 0.33)
        self.test_fraction = 1.0 - self.training_fraction - self.validation_fraction

        self.run_mode = args.get('run_mode', 'training')  # 'training' or 'inference'

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
            numpy_array = np.empty((num_events, 27))

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
                    # Dimensions: (number of events, 20)
                    features = [
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
                        # Event-level features
                        event.get('n_particles', 0),
                        mjj,
                        event.get('dR', 0.0),
                        mass_diff,
                        1.0 if label_type == 'signal' else 0.0
                    ]

                    numpy_array[event_index, :] = features
                    event_index += 1

        except Exception as e:
            print(f"Error loading data from {input_file}: {e}")
            return None

        return numpy_array
    
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
    
    def save_numpy(self, data_array, output_file):
        """
        Save numpy array to file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        np.save(output_file, data_array)

    def training_mode(self):
        """
        For training and validation, we need to create training, validation, and test sets
        """
        if not self.input_background or not self.input_signal:
            raise ValueError("Both input_background and input_signal must be provided for training mode.")
        
        background_data = self.load_to_numpy(self.input_background)
        signal_data = self.load_to_numpy(self.input_signal, label_type='signal')

        combined_data = np.vstack((background_data, signal_data))
        combined_data = self.shuffle_and_split(
            combined_data, 
            train_fraction=self.training_fraction, 
            val_fraction=self.validation_fraction
        )

        train_set, val_set, test_set = combined_data

        self.save_numpy(train_set, os.path.join(self.output_dir, 'train.npy'))
        self.save_numpy(val_set, os.path.join(self.output_dir, 'validation.npy'))
        self.save_numpy(test_set, os.path.join(self.output_dir, 'test.npy'))

        prompt_output = (
            "<tool_result>"
            f"Training, validation, and test sets created and saved.\n"
            f"Training set shape: {train_set.shape}\n"
            f"Validation set shape: {val_set.shape}\n"
            f"Test set shape: {test_set.shape}\n"
            "</tool_result>"
        )

        print(prompt_output)

    def inference_mode(self):
        """
        This is for inference mode, primarily when the data is unlabeled (real data)
        """
        if not self.input_unlabeled:
            raise ValueError("input_unlabeled must be provided for inference mode.")
        
        mixed_data = self.load_to_numpy(self.input_unlabeled)
        shuffled_data = self.shuffle(mixed_data)
        
        self.save_numpy(shuffled_data, os.path.join(self.output_dir, 'inference.npy'))

        prompt_output = (
            "<tool_result>"
            f"Inference set created and saved.\n"
            f"Inference set shape: {shuffled_data.shape}\n"
            "</tool_result>"
        )

        print(prompt_output)

    def run(self):
        if self.run_mode == 'training':
            self.training_mode()
        elif self.run_mode == 'inference':
            self.inference_mode()
        else:
            raise ValueError(f"Unknown run mode: {self.run_mode}")

if __name__ == "__main__":
    preperation = LaCATHODEPreperation(
        **vars(args)
    )
    preperation.run()