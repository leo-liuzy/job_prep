from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, model_size, max_length):
        