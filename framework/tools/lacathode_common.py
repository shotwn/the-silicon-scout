import numpy as np
import os
import sys
import lacathode_event_dictionary as LEDict
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

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
    from sk_cathode.utils.preprocessing import LogitScaler
except ImportError:
    print("Error: sk_cathode library not found. Please ensure the sk_cathode folder is in your path.")
    sys.exit(1)

# Define constants for each processor feature
# Like Standard Scale, Logit Transform, and Dequantization


class LaCATHODEProcessorFAILED:
    def __init__(self):
        self.outer_scaler = StandardScaler()
        self.latent_scaler = StandardScaler()
        self.log_offsets = {}
        self.random_seed = 42
        
        # Features used for the Flow (Must match Trainer!)
        self.use_indices = [
            LEDict.get_tag_index("j1_mass", -1),
            LEDict.get_tag_index("mass_diff", -1),
            LEDict.get_tag_index("j1_tau2_over_tau1", -1),
            LEDict.get_tag_index("j2_tau2_over_tau1", -1),
            LEDict.get_tag_index("dR", -1)
        ]

    def _apply_transforms(self, x_raw, fit=False):
        """
        Applies Dequantization -> Logit -> Log.
        fit=True: Calculates and stores the log-offsets.
        fit=False: Uses the stored log-offsets.
        """
        x = x_raw.copy()

        # 1. Dequantize (Discrete -> Continuous)
        # Shift indices by -1 because x_raw excludes the Mass column
        discrete_indices = [
            LEDict.get_tag_index("n_particles", -1),
            LEDict.get_tag_index("j1_n_particles", -1),
            LEDict.get_tag_index("j2_n_particles", -1)
        ]
        np.random.seed(self.random_seed) 
        for idx in discrete_indices:
            # Add noise [0, 1)
            x[:, idx] += np.random.uniform(0, 1.0, size=x.shape[0])

        # 2. Logit Transform (Bounded 0-1 -> Unbounded)
        ratio_indices = [
            LEDict.get_tag_index("j1_tau2_over_tau1", -1),
            LEDict.get_tag_index("j2_tau2_over_tau1", -1)
        ]
        # Clip to safe range [0.0001, 0.9999]
        x[:, ratio_indices] = np.clip(x[:, ratio_indices], 1e-4, 1 - 1e-4)
        x[:, ratio_indices] = np.log(x[:, ratio_indices] / (1 - x[:, ratio_indices]))

        # 3. Log Transform (Heavy Tails -> Normal-ish)
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
        
        for idx in log_indices:
            if fit:
                # Calculate offset to ensure positivity: log(x + offset)
                min_val = x[:, idx].min()
                offset = max(0, -min_val) + 1.0
                self.log_offsets[idx] = offset
            else:
                # Retrieve offset
                if idx not in self.log_offsets:
                    raise ValueError(f"Log offset for feature {idx} not found. You must run fit_scaler() with training data first.")
                offset = self.log_offsets[idx]
            
            x[:, idx] = np.log(x[:, idx] + offset)
            
        return x

    def fit_scaler(self, x_raw):
        """Called by Oracle/Trainer using Training Data to learn the transforms."""
        x_transformed = self._apply_transforms(x_raw, fit=True)
        # We only fit the standard scaler on the features the model actually uses
        x_selected = x_transformed[:, self.use_indices]
        self.outer_scaler.fit(x_selected)
        print("Processor fitted (Offsets and Scaler calculated).")

    def transform(self, x_raw):
        """Called during inference to transform new data."""
        x_transformed = self._apply_transforms(x_raw, fit=False)
        x_selected = x_transformed[:, self.use_indices]
        return self.outer_scaler.transform(x_selected)
    
class SafeClipper(BaseEstimator, TransformerMixin):
    """
    Helper class to clip values slightly away from 0.0 and 1.0 
    to prevent LogitScaler from producing Infinity.
    """
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Clip only the columns that will be LogitScaled (indices 2, 3, 4 typically)
        # But for safety, we can clip the whole block if it's within [0,1] range
        # Here we apply to all selected features as they are usually ratios in LHCO
        X_clipped = np.clip(X, self.epsilon, 1.0 - self.epsilon)
        return X_clipped

import numpy as np
import lacathode_event_dictionary as LEDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import numpy as np
import lacathode_event_dictionary as LEDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class LaCATHODEProcessor(BaseEstimator, TransformerMixin):
    """
    LaCATHODE Data Processor:
    - Selects relevant features
    - Applies Logit Transform to Tau21 ratios
    - Standard Scales Mass and dR features
    - Clips extreme outliers to prevent instability during model inference.
    """
    def __init__(self):
        self.outer_scaler = StandardScaler() # For input features
        self.cond_scaler = StandardScaler() # For condition (Mass)
        
        # Standard LHCO Features
        self.use_indices = [
            LEDict.get_tag_index("j1_mass", -1),           # 0: Mass (Standard)
            LEDict.get_tag_index("mass_diff", -1),         # 1: Delta Mass (Standard)
            LEDict.get_tag_index("j1_tau2_over_tau1", -1), # 2: Tau21 j1 (Logit)
            LEDict.get_tag_index("j2_tau2_over_tau1", -1), # 3: Tau21 j2 (Logit)
            LEDict.get_tag_index("dR", -1)                 # 4: Delta R (Standard)
        ]
        
        # ONLY apply Logit to the Taus (indices 2 and 3)
        self.logit_indices = [
            2, 3 # These indices are for USE_INDICES array
        ]

        self.epsilon = 1e-4

    def fit_scaler(self, x_raw, m_raw=None):
        # 1. Select
        x_selected = x_raw[:, self.use_indices]
        
        # 2. Transform (Logit Only on Taus)
        x_trans = self._apply_transforms(x_selected)
        
        # 3. Fit Scaler
        self.outer_scaler.fit(x_trans)

        # 4. Fit Condition Scaler if Mass provided
        if m_raw is not None:
            self.cond_scaler.fit(m_raw)
            print("LaCATHODE Processor fitted (Logit for Taus, Standard for Mass/dR).")
        else:
            print("LaCATHODE Processor fitted (Logit for Taus, ! No Mass Scaling).")

    def transform(self, x_raw):
        x_selected = x_raw[:, self.use_indices]
        
        # 1. Math Transform
        x_trans = self._apply_transforms(x_selected)
        
        # 2. Standard Scaling
        x_scaled = self.outer_scaler.transform(x_trans)
        
        # 3. CRITICAL: Hard Clip to prevent -15 sigma outliers from Taus
        return np.clip(x_scaled, -5.0, 5.0)
    
    def transform_condition(self, m_raw):
        """Transforms Condition: Scale -> Clip"""
        # Check if fitted, otherwise fit on the fly (safety fallback)
        try:
            m_scaled = self.cond_scaler.transform(m_raw)
        except:
            print("Warning: Condition scaler not fitted. Fitting on current batch.")
            self.cond_scaler.fit(m_raw)
            m_scaled = self.cond_scaler.transform(m_raw)
            
        return np.clip(m_scaled, -5.0, 5.0)

    def _apply_transforms(self, X):
        X_new = X.copy()
        
        # Apply Logit ONLY to Ratios [0, 1]
        for i in self.logit_indices:
            # Clip to avoid log(0) or division by zero
            X_new[:, i] = np.clip(X_new[:, i], self.epsilon, 1.0 - self.epsilon)
            X_new[:, i] = np.log(X_new[:, i] / (1.0 - X_new[:, i]))

        # DO NOT Apply Log to Mass or dR (Columns 0, 1, 4). 
        # Leave them linear so negative mass_diff doesn't crash everything.
            
        return X_new

    def sanitize(self, data, *others):
        mask = np.all(np.isfinite(data), axis=1)
        cleaned_data = data[mask]
        cleaned_others = [arr[mask] for arr in others]
        if len(cleaned_others) == 0: return cleaned_data
        return (cleaned_data, *cleaned_others)