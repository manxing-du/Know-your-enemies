import torch
import torch.nn as nn
from TransformerModels import PositionalEncoding

class PreprocessInput(nn.Module):
    def __init__(self, vocab_size, model_dim, drop_prob, is_cuda):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encode = PositionalEncoding(model_dim=model_dim, dropout=drop_prob, is_cuda=is_cuda)

    def forward(self, inp):
        embedding = self.embedding(inp)
        return self.pos_encode(embedding)
        # return embedding

