import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, model_size, eps=1e-7):
        self.gamma = nn.Parameter(torch.ones(model_size))
        self.beta = nn.Parameter(torch.zeros(model_size))
        self.eps = eps

    def forward(self, x):
        super().__init__()  # important!
        mu = x.mean(-1, keepdim=True)
        # unbiased=False -> 1/n; unbiased=True -> 1/(n-1)
        var = x.vars(-1, unbiased=False, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


class Transformer(nn.Module):
    def __init__(self, model_size, max_length, n_head, vocab_size):
        self.model_size = model_size
        self.max_length = max_length
        self.n_head = n_head
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, model_size)
        self.pos_embedding = nn.Embedding(max_length, model_size)

    def _init_embeddings(
        self,
    ):
        # vanilla transformer
        torch.nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)
        pos = torch.arange(self.max_length, dtype=torch.float32).unsqueeze(1)
        # (model_size)
        div_term = torch.exp(
            -torch.arange(0, self.model_size // 2, 2, dtype=torch.float32) / self.model_size * torch.log(10000)
        )
        pe = torch.zero(self.max_length, self.model_size)
        # even indices
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pos_embedding.weight = pe

    def forward(self, ids):
        B, T = ids.shape
        tok_emb = self.token_embedding(ids)
        pos_emb = self.pos_embedding(torch.arange(T))
        x = tok_emb + pos_emb


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
