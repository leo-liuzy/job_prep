import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, dropout=0.1):
        super().__init__()  # important!
        assert model_size % n_head == 0, "model_size must be divisible by n_head"
        self.model_size = model_size
        self.n_head = n_head
        self.head_dim = model_size // n_head
        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)

        self.out_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = self.mask[:, :, :T, :T].to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, n_head, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, head_dim)
        out = attn @ V
        # (B, T, model_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, n_group=1, dropout=0.1):
        super().__init__()  # important!
        assert model_size % n_head == 0, "model_size must be divisible by n_head"
        # Difference
        assert n_head % n_group == 0, "n_head must be divisible by n_group"

        self.model_size = model_size
        self.n_head = n_head
        self.n_group = n_group
        self.head_dim = model_size // n_head
        # Difference
        self.heads_per_group = n_head // n_group

        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.out_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Difference: : [B, n_group, T, head_dim]
        K = K.view(B, T, self.n_group, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_group, self.head_dim).transpose(1, 2)

        # Difference: n_head = heads_per_group * n_group
        K = K.repeat_interleave(self.heads_per_group, dim=1)  # expand K per head
        V = V.repeat_interleave(self.heads_per_group, dim=1)  # expand V per head

        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = self.mask[:, :, :T, :T].to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, n_head, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, head_dim)
        out = attn @ V
        # (B, T, model_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class MultiQueryAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, dropout=0.1):
        super().__init__()  # important!
        assert model_size % n_head == 0, "model_size must be divisible by n_head"
        self.model_size = model_size
        self.n_head = n_head
        self.head_dim = model_size // n_head
        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(
            model_size,
        )
        self.v_proj = nn.Linear(model_size, model_size)

        self.out_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape into heads: [B, 1, T, head_dim]
        #### Difference ####
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        #### Difference ####

        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = self.mask[:, :, :T, :T].to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, n_head, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, head_dim)
        out = attn @ V
        # (B, T, model_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, model_size, max_length, n_head, dropout=0.1):
        super().__init__()  # important!
        assert model_size % n_head == 0, "model_size must be divisible by n_head"
        self.model_size = model_size
        self.n_head = n_head
        self.head_dim = model_size // n_head
        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)

        self.out_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        ##### Difference ######
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)  # concat along sequence dimension
            V = torch.cat([V_prev, V], dim=2)

        kv_cache_new = (K, V)
        seq_len = K.shape[2]
        mask = self.mask[:, :, :T, :seq_len].to(x.device)
        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        if kv_cache is not None:
            # Current tokens are at positions [seq_len-T : seq_len]
            # They should attend to all previous positions [0 : seq_len-T+i] for token i
            start_pos = seq_len - T
            mask = self.mask[:, :, start_pos:seq_len, :seq_len].to(x.device)
        else:
            # Standard case: no cache, just use first T rows and columns
            mask = self.mask[:, :, :T, :T].to(x.device)

        ##### Difference ######
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, n_head, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, head_dim)
        out = attn @ V
        # (B, T, model_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, kv_cache_new
