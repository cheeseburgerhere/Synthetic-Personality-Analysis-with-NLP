import torch
import torch.nn as nn
import torch.nn.functional as F
# 1. Define the Projector Network
class HobbyProjector(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, output_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        # Normalize output to keep it on the hypersphere (important for cosine similarity)
        # However, if the target vectors are not normalized, we might strictly want regression.
        # Usually embeddings are compared via Cosine Similarity, so normalizing is good practice.
        return torch.nn.functional.normalize(self.net(x), p=2, dim=1)