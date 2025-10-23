import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, model_size):
        self.gamma = nn.Parameter(torch.ones(1, 1, model_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, model_size))

    def forward(self, x):
        # B x T x H
        mu = x.mean(-1, keep_dim=True)
        std = x.std(-1, keep_dim=True, unbiased=False)
        x = (x - mu) / std
        return self.gamma * x + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, num_head, dropout=0.1):
        self.model_size = model_size
        self.num_head = num_head

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.o_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)
