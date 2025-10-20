from torch import nn
import torch
import numpy as np
from transformers.data.data_collator import default_data_collator

default_precision_type = torch.float16

class NumericFusionAdapter(nn.Module):
    def __init__(self, hidden_size, numeric_dim, dtype=default_precision_type, device=None):
        super().__init__()
        # Project numeric features to hidden size
        self.fc = nn.Linear(numeric_dim, hidden_size, dtype=dtype, device=device)

        # Added: Because text embeddings are typically small in magnitude
        # Normalize to match token embedding scale
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        # Scale to match token embedding scale
        self.scale = nn.Parameter(torch.tensor(0.001))  # match token embedding scale

    def forward(self, numeric_features):
        # Project numeric features to same dim as embeddings
        # Ensure dtype matches model
        x = numeric_features.to(self.fc.weight.dtype)
        # Linear projection
        x = self.fc(x)
        # Normalize
        x = self.norm(x)
        # Scale to match token embedding scale
        # Tan h makes sure values are in [-1, 1], then rescale
        x = torch.tanh(x) * self.scale  # compress + rescale
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

        return batch