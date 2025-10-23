import torch
from torch import nn


class PositionEmbedding(nn.Module):
    # Vanilla Transformer
    def __init__(self, model_size, max_length):
        self.model_size = model_size
        self.max_length = max_length
        # (max_len, 1)
        self.pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # (model_size)
        div_term = torch.exp(
            -torch.arange(0, model_size // 2, 2, dtype=torch.float32) / self.model_size * torch.log(10000)
        )
        pe = torch.zero(max_length, model_size)

        self.weight = 10000 ** (self.dim / self.model_size)

        import pdb

        pdb.set_trace()
        10000 ** (self.dim / self.model_size)


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
