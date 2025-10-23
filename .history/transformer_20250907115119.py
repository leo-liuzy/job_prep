import torch 
from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, model_size, max_length):
        self.
        self.pos = torch.arange(model_size)