import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, model_size, eps=1e-7):
        super().__init__()  # important!
        self.gamma = nn.Parameter(torch.ones(model_size))
        self.beta = nn.Parameter(torch.zeros(model_size))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        # unbiased=False -> 1/n; unbiased=True -> 1/(n-1)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


class FeedForward(nn.Module):
    def __init__(self, model_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_size, model_size * 4), nn.ReLU(), nn.Linear(model_size * 4, model_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


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


class CrossMHAttention(nn.Module):
    def __init__(
        self,
        model_size,
        num_head,
        dropout=0.1,
    ):
        super().__init__()
        self.model_size = model_size
        self.num_head = num_head
        assert self.model_size % self.num_head == 0
        self.head_size = self.model_size // self.num_head

        # projections for Q (from decoder), K and V (from encoder)
        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.o_proj = nn.Linear(model_size, model_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_mask=None):
        """
        x: (B, T_q, D)   → decoder input embeddings (queries)
        context: (B, T_kv, D) → encoder hidden states (keys/values)
        context_mask: (B, 1, 1, T_kv) optional mask for padding in encoder outputs
        """
        B, T_q, D = x.shape
        T_kv = context.size(1)

        Q = self.q_proj(x)  # (B, T_q, D)
        K = self.k_proj(context)  # (B, T_kv, D)
        V = self.v_proj(context)  # (B, T_kv, D)

        # reshape into heads
        Q = Q.view(B, T_q, self.num_head, self.head_size).transpose(1, 2)  # (B, H, T_q, d_h)
        K = K.view(B, T_kv, self.num_head, self.head_size).transpose(1, 2)  # (B, H, T_kv, d_h)
        V = V.view(B, T_kv, self.num_head, self.head_size).transpose(1, 2)  # (B, H, T_kv, d_h)

        # scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.head_size**0.5)  # (B, H, T_q, T_kv)

        if context_mask is not None:  # typically mask out encoder padding positions
            scores = scores.masked_fill(context_mask == 0, -float("inf"))

        attn = scores.softmax(dim=-1)  # (B, H, T_q, T_kv)
        attn = self.dropout(attn)
        output = attn @ V  # (B, H, T_q, d_h)

        # combine heads
        output = output.transpose(1, 2).contiguous().view(B, T_q, D)  # (B, T_q, D)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output


class TransformerLayer(nn.Module):
    # Decoder Only model
    def __init__(self, model_size, max_length, n_head, dropout):
        super().__init__()  # important!
        self.attn = MultiHeadAttention(model_size, max_length, n_head, dropout)
        self.ffn = FeedForward(model_size, dropout)
        self.ln1 = LayerNorm(model_size)
        self.ln2 = LayerNorm(model_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_size,
        num_head,
        max_length,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        super().__init__()
        # (1) Self-attention (bidirectional, not causal)
        self.self_attn = MultiHeadAttention(
            model_size,
            num_head,
            max_length,
            dropout,
        )
        # (2) Feedforward
        self.feedforward = FeedForward(model_size, intermediate_size_ratio, dropout)

        # LayerNorms
        self.norm1 = LayerNorm(model_size)  # for self-attn
        self.norm2 = LayerNorm(model_size)  # for FFN

    def forward(self, x, src_mask=None):
        """
        x: (B, T_enc, D) input sequence embeddings
        src_mask: (B, 1, 1, T_enc) optional padding mask
        """
        # (1) self-attention (full sequence, not causal)
        x = x + self.self_attn(self.norm1(x), mask=src_mask)

        # (2) feedforward
        x = x + self.feedforward(self.norm2(x))

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        model_size,
        num_head,
        max_length,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        super().__init__()
        # (1) Decoder self-attention (causal)
        self.self_attn = MultiHeadAttention(
            model_size,
            num_head,
            max_length,
            dropout,
        )
        # (2) Cross-attention (attends to encoder outputs)
        self.cross_attn = CrossMHAttention(
            model_size,
            num_head,
            dropout,
        )
        # (3) Feedforward
        self.feedforward = FeedForward(model_size, intermediate_size_ratio, dropout)

        # LayerNorms
        self.norm1 = LayerNorm(model_size)  # for self-attn
        self.norm2 = LayerNorm(model_size)  # for cross-attn
        self.norm3 = LayerNorm(model_size)  # for FFN

    def forward(self, x, encoder_out, encoder_mask=None):
        """
        x: (B, T_dec, D)      decoder input
        encoder_out: (B, T_enc, D) encoder outputs
        encoder_mask: (B, 1, 1, T_enc) optional padding mask for encoder
        """
        # (1) masked self-attention
        x = x + self.self_attn(self.norm1(x))

        # (2) cross-attention with encoder outputs
        x = x + self.cross_attn(self.norm2(x), encoder_out, encoder_mask)

        # (3) feedforward
        x = x + self.feedforward(self.norm3(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        model_size,
        num_head,
        max_length,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    model_size,
                    num_head,
                    max_length,
                    intermediate_size_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(model_size)  # final layer norm

    def forward(self, x, src_mask=None):
        """
        x: (B, T_enc, D)
        src_mask: (B, 1, 1, T_enc)
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        model_size,
        num_head,
        max_length,
        intermediate_size_ratio,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    model_size,
                    num_head,
                    max_length,
                    intermediate_size_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(model_size)  # final layer norm

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        """
        x: (B, T_dec, D)
        encoder_out: (B, T_enc, D)
        src_mask: (B, 1, 1, T_enc) mask for encoder (padding)
        tgt_mask: (B, 1, T_dec, T_dec) causal mask for decoder self-attention
        """
        for layer in self.layers:
            # pass tgt_mask into decoder self-attention
            x = layer(x, encoder_out, encoder_mask=src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        model_size,
        max_length,
        vocab_size,
        n_layer,
        n_head,
    ):
        super().__init__()  # important!
        self.model_size = model_size
        self.max_length = max_length
        self.n_head = n_head
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, model_size)
        self.pos_embedding = nn.Embedding(max_length, model_size)

        self.blocks = nn.Sequential(*[TransformerLayer(model_size, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = LayerNorm(model_size)
        self.lm_head = nn.Linear(model_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _init_embeddings(
        self,
    ):
        # vanilla transformer
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        pos = torch.arange(self.max_length, dtype=torch.float32).unsqueeze(1)
        # (model_size)
        div_term = torch.exp(
            -torch.arange(0, self.model_size, 2, dtype=torch.float32) / self.model_size * torch.log(10000)
        )
        pe = torch.zeros(self.max_length, self.model_size)
        # even indices
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pos_embedding.weight = pe

    def forward(self, ids):
        B, T = ids.shape
        tok_emb = self.token_embedding(ids)
        pos_emb = self.pos_embedding(torch.arange(T)).unsqueeze(0)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        labels = ids[:, 1:]
        loss_logits = logits[:, :-1, :]

        loss = F.cross_entropy(loss_logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        return logits, loss


if __name__ == "__main__":
    PE = PositionEmbedding(100, 10000)
