import torch
import torch.nn as nn

class PointWiseFeedForward(nn.Module):
    def __init__(self, model_dim, between_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(model_dim, between_dim),
            nn.ReLU(True),
            nn.Linear(between_dim, model_dim)
        )

    def forward(self, inp):
        return self.layers(inp)
