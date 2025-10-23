import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, model_size):
        self.gamma = nn.Parameter(torch.ones(1, 1, model_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, model_size))
