import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim_k, model_dim):
        super().__init__()
        self.dim_k = dim_k

        self.weight_query = nn.Linear(model_dim, dim_k, bias=False)
        self.weight_key = nn.Linear(model_dim, dim_k, bias=False)
        self.weight_value = nn.Linear(model_dim, dim_k, bias=False)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.weight_query(query), self.weight_key(key), self.weight_value(value)
        pre_score = torch.matmul(query, torch.transpose(key, -1, -2)) / math.sqrt(self.dim_k)

        if mask is not None:
            pre_score = pre_score.masked_fill(mask == 0, -1e9)

        score = F.softmax(pre_score, dim=-1)
        return torch.matmul(score, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_head):
        super().__init__()
        assert model_dim%num_head == 0, "The model dimension should be in multiple of the number of heads"

        self.model_dim = model_dim
        self.num_head = num_head
        self.dim_all = model_dim//num_head

        self.weight_output = nn.Linear(model_dim, model_dim, bias=False)
        self.all_attention_head = nn.ModuleList([Attention(self.dim_all, model_dim) for _ in range(num_head)])

    def forward(self, query, key, value, mask=None):
        all_heads = [
            head(query, key, value, mask=mask)
            for head in self.all_attention_head
        ]
        concat_heads = torch.cat(all_heads, -1)
        return self.weight_output(concat_heads)
