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
