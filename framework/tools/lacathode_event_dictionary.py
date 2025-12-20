tags = [
    "mass",               # 0: Invariant mass of the dijet system (mjj); used as the conditional variable.
    "n_particles",        # 1: Total number of particles detected across the entire dijet event.
    "dR",                 # 2: Angular distance (Delta R) between the axes of the two leading jets.
    "mass_diff",          # 3: The absolute difference in mass between the first and second jet (|mj1 - mj2|).
    
    # --- Leading Jet (Jet 1) Features ---
    "j1_px",              # 4: X-component of the momentum for the leading jet.
    "j1_py",              # 5: Y-component of the momentum for the leading jet.
    "j1_pz",              # 6: Z-component of the momentum for the leading jet.
    "j1_mass",            # 7: Invariant mass of the leading jet (mj1).
    "j1_n_particles",     # 8: Multiplicity (number of constituents/particles) within the leading jet.
    "j1_P_T_lead",        # 9: Transverse momentum (pT) of the highest-momentum particle inside Jet 1.
    "j1_tau_1",           # 10: 1-subjettiness; measures how much the jet radiation is aligned along one axis.
    "j1_tau_2",           # 11: 2-subjettiness; measures alignment along two sub-axes.
    "j1_tau_3",           # 12: 3-subjettiness; measures alignment along three sub-axes.
    "j1_tau_4",           # 13: 4-subjettiness; measures alignment along four sub-axes.
    "j1_tau2_over_tau1",  # 14: N-subjettiness ratio (tau21); used to distinguish 2-pronged from 1-pronged jets.

    # --- Subleading Jet (Jet 2) Features ---
    "j2_px",              # 15: X-component of the momentum for the subleading jet.
    "j2_py",              # 16: Y-component of the momentum for the subleading jet.
    "j2_pz",              # 17: Z-component of the momentum for the subleading jet.
    "j2_mass",            # 18: Invariant mass of the subleading jet (mj2).
    "j2_n_particles",     # 19: Multiplicity (number of constituents/particles) within the subleading jet.
    "j2_P_T_lead",        # 20: Transverse momentum (pT) of the highest-momentum particle inside Jet 2.
    "j2_tau_1",           # 21: 1-subjettiness for the subleading jet.
    "j2_tau_2",           # 22: 2-subjettiness for the subleading jet.
    "j2_tau_3",           # 23: 3-subjettiness for the subleading jet.
    "j2_tau_4",           # 24: 4-subjettiness for the subleading jet.
    "j2_tau2_over_tau1",  # 25: N-subjettiness ratio (tau21) for the subleading jet.

    "label"               # 26: Ground truth identity; 1.0 for Signal events, 0.0 for Background events.
]

def get_tag_index(tag_name, shift=0):
    """Returns the index of the given tag name."""
    try:
        return tags.index(tag_name) + shift
    except ValueError:
        raise ValueError(f"Tag '{tag_name}' not found in the event dictionary.")