import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, model_size, max_length):
        self.model_size = model_size
        self.max_length = max_length
        self.pos = torch.arange(max_length)
        self.dim = (torch.arange(model_size) // 2).int()
        self.dim = self.dim.unsqueeze(0).repeat(max_length, 1)
        10000 ** (self.dim / self.model_size)


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
