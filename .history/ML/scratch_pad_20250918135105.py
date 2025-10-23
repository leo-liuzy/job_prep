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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_size,
        num_head,
        max_length,
        dropout=0.1,
    ):
        super().__init__()
        self.model_size = model_size
        self.num_head = num_head
        assert self.model_size % self.num_head == 0
        self.head_size = self.model_size // self.num_head
        self.mask = nn.register_buffer("mask", torch.tril(max_length, max_length).unsqueeze(0).unsqueeze(0))

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.o_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.num_head, self.head_size).transpose(1, 2)

        mask = self.mask[:, :, :T, :T]
        scores = (Q @ K) / (self.head_size**0.5)
        scores = scores.masked_fill(mask == 0, -float("inf"))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        output = attn @ V

        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output


class Feedfoward(nn.Module):
    def __init__(
        self,
        model_size,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        super().__init__()
        self.intermediate_size_ratio = intermediate_size_ratio
        self.intermediate_size = int(intermediate_size_ratio * model_size)
        self.fc1 = nn.Linear(model_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, model_size)
        self.activation = nn.GeLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(x):
    def __init__(
        self,
        model_size,
        num_head,
        max_length,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        self.attention = MultiHeadAttention(
            model_size,
            num_head,
        )
