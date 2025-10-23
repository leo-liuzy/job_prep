import torch
from torch import nn


class PositionEmbedding(nn.Module):
    # Vanilla Transformer
    def __init__(self, model_size, max_length):
        self.model_size = model_size
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, model_size)
        # (max_len, 1)
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # (model_size)
        div_term = torch.exp(
            -torch.arange(0, model_size // 2, 2, dtype=torch.float32) / self.model_size * torch.log(10000)
        )
        pe = torch.zero(max_length, model_size)
        # even indices
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.embedding.weight = pe

    def __forward__(self, ids):
        B, T = ids.shape
        pos_embeddings = self.embedding(torch.arange(T, device=ids.device))

        # return self.embedding(ids)


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
        torch.nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # (model_size)
        div_term = torch.exp(
            -torch.arange(0, model_size // 2, 2, dtype=torch.float32) / self.model_size * torch.log(10000)
        )
        pe = torch.zero(max_length, model_size)
        # even indices
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
