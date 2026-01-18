import pandas as pd
import numpy as np 
import vector
import awkward as awk
import fastjet

vector.register_awkward()

import json
import os

# Load the DataFrame from the HDF5 file
file_path = "events_anomalydetection_v2.h5"
# file_path = "events_anomalydetection_tiny.h5" # For testing with smaller dataset
# file_path = "events_LHCO2020_BlackBox1.h5"
numpy_read_chunk_size = 100000  # Number of rows to read at a time

size_per_row = 2100  # 2100 data + 1 label will make 2101 columns
no_label = False # Disables labels, all data outputs as background
# blackbox_label_file = "events_LHCO2020_BlackBox1.masterkey"  # Whether to use external blackbox labels
blackbox_label_file = None  # Whether to use external blackbox labels

# Load blackbox labels if provided
blackbox_labels = None
if blackbox_label_file and not no_label:
    with open(blackbox_label_file, "r") as f:
        # Blackbox labels are simple ASCII text files with one label per line
        blackbox_labels = f.read().splitlines()


def get_is_signal_label_from_event(event_order_index, event):
    if no_label:
        return None
    
    if blackbox_labels:
        return True if blackbox_labels[event_order_index] == "1.0" else False
    else:
        return event[size_per_row]  # Last column is the label. Notice this will look for (size_per_row + 1)st item in 0-indexing

def process_events_chunk(df, start_index=0):
    # Store jets seperately depending on whether they are signal or background
    # Store jets in memory and write to file every 5000 events to save RAM
    output_file_cache_size = 5000
    signal_events = []
    background_events = []
    process_id = os.getpid()

    def write_events_to_file(events, type):
        if type == "signal":
            file_instance = open(f"output/signal_events_{process_id}.jsonl", "a")
        else:
            file_instance = open(f"output/background_events_{process_id}.jsonl", "a")

        chunk = [{'type': type, **(event)} for event in events]
        for event in chunk:
            file_instance.write(json.dumps(event) + "\n")
            file_instance.flush()

        print(f"Wrote {len(chunk)} events to file.")
        file_instance.close()

    # Now process the DataFrame chunk
    events_combined = df.T # Transpose the DataFrame from [cols, rows] to [rows, cols] so events_combined[i] is the i-th event
    
    # Loop through each event (row) in the transposed DataFrame
    for i in range(start_index, np.shape(events_combined)[1] + start_index):
        # This data set includes both background and signal events
        # Each row of events_combined is a collision event
        # Index 2100 is a tag or label that indicates whether the event is background (0) or signal (1)
        # Background events are normal events, while signal events are rare events that may indicate new physics phenomena
        # Carried this logic to a function to handle no_label and blackbox labels
        is_signal = get_is_signal_label_from_event(i, events_combined[i])

        # Data starts as background and ends as signals, events 1098978 to 1099999 are signals
        event_data = events_combined[i][:size_per_row] # Get the event data excluding the label

        """
        From LHC Olympics 2020 website:
        Both the “Simulation” and “Data” have the following event selection: 
        at least one anti-kT R = 1.0 jet with pseudorapidity |η| < 2.5 and transverse momentum pT > 1.2 TeV.
        For each event, we provide a list of all hadrons (pT, η, φ, pT, η, φ, …) zero-padded up to 700 hadrons.
        """

        # So for events_anomalydetection_v2.h5 each particle has 3 numbers: pT, eta, phi and rows are padded to 700 particles totalling on 2100 columns + 1 label column
        # [pT₁, η₁, φ₁,  pT₂, η₂, φ₂,  pT₃, η₃, φ₃,  ..., pT₇₀₀, η₇₀₀, φ₇₀₀, label]
        # For pT take every 3rd element starting from index 0

        # Create an empty array for pseudojets input
        # We use only the non-zero pT values to determine the length
        # Numpy is needed because vanilla python fills RAM too quickly
        awkward_builder = awk.ArrayBuilder()

        # Fill the pseudojets input array with pT, eta, phi values
        for j in range(len([x for x in event_data if x != 0]) // 3):
            with awkward_builder.record():
                awkward_builder.field("pt").append(event_data[j * 3])
                awkward_builder.field("eta").append(event_data[j * 3 + 1])
                awkward_builder.field("phi").append(event_data[j * 3 + 2])
                awkward_builder.field("M").append(0.0)  # Assuming massless particles


        # Combine pT, eta, phi into a list of jets for the event
        # This is from cluster library, which I did not use before
        # R is the jet radius parameter, p is the algorithm parameter (p=-1 for anti-kt)
        # ptmin is the minimum transverse momentum for a jet to be considered
        # jets = cluster(pseudojets_input, R=1.0, p=-1).inclusive_jets(ptmin=1200.0) # Anti-kt algorithm with R=1.0, p=-1 for anti-kt, ptmin=1200 GeV
        # ! pyjet is now unmaintained, use fastjet instead

        jets_input_array = awk.Array(awkward_builder.snapshot(), with_name="Momentum4D")

        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
        cluster = fastjet.ClusterSequence(jets_input_array, jetdef)

        inclusive_jets = cluster.inclusive_jets(min_pt=1200.0)
        constituent_array = cluster.constituents(min_pt=1200.0)
        

        # max_PT_indices[i] -> jets[i]'s leading constituent index in constituent_array
        max_PT_indices = awk.argmax(constituent_array.pt, axis=1)
        
        jets = []
        invariant_mass = np.float64(0.0)
        for j, jet in enumerate(inclusive_jets):
            jets.append({
                "P_T": jet.pt,
                "eta": jet.eta,
                "phi": jet.phi,
                "E": jet.E,
                "m": jet.m,
                "n_particles": len(constituent_array[j]),
                # Leading constituent
                "P_T_lead": constituent_array[j][max_PT_indices[j]].pt,
                # Delta R with other jets will be calculated later on
                "dR": {}
            })
            invariant_mass += jet.m

            # O^2 op.
            for k, other_jet in enumerate(inclusive_jets):
                if j != k:
                    # Calculate delta R between jets
                    dR = jet.deltaR(other_jet)
                    jets[j]["dR"][f"jet{k+1}"] = dR

        additional_info = {
            "n_particles": cluster.n_particles(),
            "M_jj": invariant_mass, # Invariant mass of the dijet system
        }

        events_list_to_append = signal_events if is_signal == 1 else background_events

        events_list_to_append.append(awk.to_list({
            "jets": jets,
            **additional_info
        }))

        if len(events_list_to_append) >= output_file_cache_size:  # Write to file every 1000 events to save RAM
            write_events_to_file(events_list_to_append, "signal" if is_signal == 1 else "background")
            events_list_to_append.clear()
    
    # Write any remaining jets to file
    write_events_to_file(signal_events, "signal")
    write_events_to_file(background_events, "background")


def generator():
    i=0
    while True:
        yield pd.read_hdf(file_path, start=i*numpy_read_chunk_size, stop=(i+1)*numpy_read_chunk_size)

        i+=1

import multiprocessing
num_cpus = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cpus}")

if __name__ == "__main__":
    # Create output dir if not exists
    if not os.path.exists("output"):
        os.makedirs("output")
    else:
        # Clear output dir
        for file in os.listdir("output"):
            os.remove(os.path.join("output", file))
            
    gen = generator()
    chunk_index = 0
    procs: list[multiprocessing.Process] = []
    while True:
        try:
            df_chunk = next(gen)
            if df_chunk.empty:
                break

            p = multiprocessing.Process(target=process_events_chunk, args=(df_chunk, chunk_index))
            p.start()
            procs.append(p)
            # If we have too many processes, wait for them to finish
            if len(procs) >= num_cpus:
                for proc in procs:
                    proc.join(timeout=6400) # Wait for all processes to finish
                procs = []

            chunk_index += numpy_read_chunk_size
            print(f"Started processing chunk {chunk_index // numpy_read_chunk_size}")

        except ValueError:
            print("Error processing chunk")
            break
    
    # Wait for any remaining processes to finish
    for proc in procs:
        proc.join()
    
    # After all chunks are processed, merge the output files
    background_file_format = "output/background_events_*.jsonl"
    signal_file_format = "output/signal_events_*.jsonl"

    import glob
    background_files = glob.glob(background_file_format)
    signal_files = glob.glob(signal_file_format)

    with open("output/background_events.jsonl", "w") as outfile:
        for fname in background_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname) # Remove the chunk file after merging
    with open("output/signal_events.jsonl", "w") as outfile:
        for fname in signal_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname) # Remove the chunk file after merging
