import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, model_size, eps=1e-7):
        super().__init__()  # important!
        self.gamma = nn.Parameter(torch.ones(model_size))
        self.beta = nn.Parameter(torch.zeros(model_size))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        # unbiased=False -> 1/n; unbiased=True -> 1/(n-1)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


class BatchNorm(nn.Module):
    def __init__(self, model_size, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_size))
        self.beta = nn.Parameter(torch.zeros(model_size))

    def forward(self, x):
        # B, T, H
        mu = x.mean(dim=(0, 1), keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.gamma + self.beta
