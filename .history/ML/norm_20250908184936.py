import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, model_size, eps=1e-5):
        super().__init__()
        self.eps
        self.gamma = nn.Parameter(torch.ones(1, 1, model_size))

    def forward(self, x):
        # B, T, H
        rms = torch.sqrt(torch.mean(x**2, keepdim=True, dim=-1) + self.eps)
        return x / rms * self.gamma


class LayerNorm(nn.Module):
    def __init__(self, model_size, eps=1e-7):
        super().__init__()  # important!
        self.gamma = nn.Parameter(torch.ones(1, 1, model_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, model_size))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        # unbiased=False -> 1/n; unbiased=True -> 1/(n-1)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


class BatchNorm(nn.Module):
    def __init__(self, model_size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, model_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, model_size))
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(1, 1, model_size))
        self.register_buffer("running_var", torch.ones(1, 1, model_size))

    def forward(self, x):
        # B, T, H
        # Update running statistics
        if self.training:
            mu = x.mean(dim=(0, 1), keepdim=True)
            var = x.var(dim=(0, 1), unbiased=False, keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu = self.running_mean
            var = self.running_var

        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.gamma + self.beta
