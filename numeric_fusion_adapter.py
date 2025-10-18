from torch import nn

class NumericFusionAdapter(nn.Module):
    def __init__(self, hidden_size, numeric_dim):
        super().__init__()
        self.fc = nn.Linear(numeric_dim, hidden_size)

    def forward(self, numeric_features):
        # Project numeric features to same dim as embeddings
        return self.fc(numeric_features).unsqueeze(1)  # shape: (B, 1, hidden_size)