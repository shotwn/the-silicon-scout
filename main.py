import pandas as pd
import numpy as np 
from fastjet import cluster, DTYPE_PTEPM

# Load the DataFrame from the HDF5 file
df = pd.read_hdf("events_anomalydetection_v2.h5", key="df")
events_combined = df.T # Transpose the DataFrame

# This data set includes both background and signal events
# Each row of events_combined is a collision event
# Index 2100 is a tag or label that indicates whether the event is background (0) or signal (1)
# Background events are normal events, while signal events are rare events that may indicate new physics phenomena

# Store jets seperately depending on whether they are signal or background
signal_jets = []
background_jets = []

# Loop through each event (row) in the transposed DataFrame
for i in range(np.shape(events_combined)[1]):
    is_signal = events_combined[i][2100] # Get the label for the event
    # Data starts as background and ends as signals, events 1098978 to 1099999 are signals

    """
    From LHC Olympics 2020 website:
    Both the “Simulation” and “Data” have the following event selection: 
    at least one anti-kT R = 1.0 jet with pseudorapidity |η| < 2.5 and transverse momentum pT > 1.2 TeV.
    For each event, we provide a list of all hadrons (pT, η, φ, pT, η, φ, …) zero-padded up to 700 hadrons.
    """

    # So for events_anomalydetection_v2.h5 each particle has 3 numbers: pT, eta, phi and rows are padded to 700 particles totalling on 2100 columns + 1 label column
    # [pT₁, η₁, φ₁,  pT₂, η₂, φ₂,  pT₃, η₃, φ₃,  ..., pT₇₀₀, η₇₀₀, φ₇₀₀, label]
    # For pT take every 3rd element starting from index 0
    pT = events_combined[i][::3]
    eta = events_combined[i][1::3]
    phi = events_combined[i][2::3]

    # Create an empty array for pseudojets input
    # We use only the non-zero pT values to determine the length
    # Numpy is needed because vanilla python fills RAM too quickly
    pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)

    # Fill the pseudojets input array with pT, eta, phi values
    for j in range(len(pseudojets_input)):
        pseudojets_input[j]['pT'] = pT[j]
        pseudojets_input[j]['eta'] = eta[j]
        pseudojets_input[j]['phi'] = phi[j]
        pseudojets_input[j]['mass'] = 0.0 # Assume massless particles   


    # Combine pT, eta, phi into a list of jets for the event
    # This is from cluster library, which I did not use before
    # R is the jet radius parameter, p is the algorithm parameter (p=-1 for anti-kt)
    # ptmin is the minimum transverse momentum for a jet to be considered
    # jets = cluster(pseudojets_input, R=1.0, p=-1).inclusive_jets(ptmin=1200.0) # Anti-kt algorithm with R=1.0, p=-1 for anti-kt, ptmin=1200 GeV
    # ! pyjet is now unmaintained, use fastjet instead

    # fastjet wants awkward arrays
    
    jets = cluster(pseudojets_input, R=1.0, p=-1).exclusive_jets(1) # Get the hardest jet
    if is_signal == 1:
        signal_jets.append(jets)
    else:
        background_jets.append(jets)

# Print the number of signal and background events
print(f"Number of signal events: {len(signal_jets)}")
print(f"Number of background events: {len(background_jets)}")