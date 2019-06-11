import torch
import torch.nn as nn

class InBetweenSubLayer(nn.Module):
    def __init__(self, model_dim, drop_prob, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(model_dim))
        self.b = nn.Parameter(torch.zeros(model_dim))

        self.eps = eps
        self.drop_prob = drop_prob

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, from_layer, input_layer):
        from_layer = self.dropout(from_layer)
        residual = input_layer + from_layer

        mean = residual.mean(-1, keepdim=True)
        std = residual.std(-1, keepdim=True)

        return self.a * (residual - mean) / (std + self.eps) + self.b
