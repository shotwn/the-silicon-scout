from torch import nn
import torch
import numpy as np
from transformers.data.data_collator import default_data_collator

default_precision_type = torch.float32 # ! Changed to float32 for testing

class NumericFusionAdapter(nn.Module):
    def __init__(self, hidden_size, numeric_dim, dtype=default_precision_type, device=None):
        super().__init__()
        # Project numeric features to hidden size
        self.mlp = nn.Sequential(
            # Three-layer MLP with SiLU activation and LayerNorm

            # Size is hidden_size // 2 for the intermediate layer to prevent overfitting
            # First layer
            nn.Linear(numeric_dim, hidden_size // 2, dtype=dtype, device=device), # Compressed intermediate size

            # Use SiLU activation
            # SiLU is smooth and non-linear, helps model complex relationships
            # Also keeps negative values unlike ReLU
            nn.SiLU(), # Activation 1 -> hidden_size // 2

            # Dropout for regularization
            # Dropout prevents overfitting, it mutes some activations randomly during training
            # Low value of 0.05 to avoid losing too much information
            nn.Dropout(0.05),

            # Second layer
            nn.Linear(hidden_size // 2, hidden_size, dtype=dtype, device=device), # Expand to hidden_size

            # Another SiLU activation
            nn.SiLU(), # Activation 2 -> hidden_size

            ## Final linear layer to project to hidden_size
            nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device), # Project to hidden_size, refinement
            
            # LayerNorm
            nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        )
        # Scale to match token embedding scale
        # self.scale = nn.Parameter(torch.tensor(0.001))  # match token embedding scale

    def forward(self, numeric_features):
        # Project numeric features to same dim as embeddings
        # Ensure dtype matches model
        x = numeric_features.to(self.mlp[0].weight.dtype)
        x = self.mlp(x)
        # Scale to match token embedding scale
        # x = x * self.scale # rescale
        return x.unsqueeze(1)  # shape: (B, 1, hidden_size)
    
class NumericFeatureCollator:
    def __init__(self, dtype=default_precision_type):
        self.dtype = dtype

    def __call__(self, features):
        # Extract numeric features
        numeric_data = [f.pop("numeric_features") for f in features]

        # Collate the rest
        batch = default_data_collator(features)

        # Convert numeric_data safely: first float32, then cast to desired dtype
        # Python doesn't have float16 type, so direct conversion does lead to issues
        # Convert numeric_data safely via numpy first 
        # Torch's direct float16 conversion screwed the pooch
        numeric_array = np.array(numeric_data, dtype=np.float32)  # shape: (B, numeric_dim)
        numeric_tensor = torch.from_numpy(numeric_array)

        # Cast to desired dtype
        if self.dtype == torch.float16:
            numeric_tensor = numeric_tensor.half()

        numeric_tensor = numeric_tensor.to(batch["input_ids"].device)

        #! This was missing previously
        batch["numeric_features"] = numeric_tensor  # shape: (B, numeric_dim)

        return batch