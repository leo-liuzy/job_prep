import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, model_size, max_length):
        self.model_size = model_size
        self.max_length = max_length
        self.pos = torch.arange(max_length)
        self.dim = torch.arange(model_size).unsqueeze(0).repeat(, 1)
