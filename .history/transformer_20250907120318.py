import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, model_size, max_length):
        self.model_size = model_size
        self.max_length = max_length
        # (max_len, 1)
        self.pos = torch.arange(max_length).unsqueeze(1)
        # (max_len, model_size)
        self.dim = (torch.arange(model_size) // 2).int()
        self.dim = self.dim.unsqueeze(0).repeat(max_length, 1)
        self.weight = (10000 ** (self.dim / self.model_size)) * self.pos

        import pdb

        pdb.set_trace()
        10000 ** (self.dim / self.model_size)


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
