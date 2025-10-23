import torch


head_dim = 8
theta_base = 10000
seq_len = 10

# (seq_len, head_dim/2
# [m_1*theta_1 m_1*theta_2 ... m_1*theta_head_dim/2]
# ...


def precompute_inverse_frequency_matrix(dim, theta_base=10000, max_seq_len: int = 4096):
    # 1 / (10000 ^ (2i /d)); i = 0, ..., head_dim/2
    theta = 1.0 / (theta_base ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    ))

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    seq_idx = torch.arange(max_seq_len, dtype=theta.dtype, device=theta.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()

    # cache includes both the cos and sin components and so the output shape is
    # [max_seq_len, dim // 2, 2]
    inv_freq = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
    # [max_seq_len, dim, 2]
    inv_freq = torch.repeat_interleave(inv_freq, repeats=2, dim=-2)

    return inv_freq


def rotate_half(x):
    x_shape = x.shape
    even_x = x[..., ::2]
    odd_x = x[..., 1::2]
    x = torch.stack([-odd_x, even_x], dim=-1).view(*x_shape[:-1], -1)
    return x


def apply_rotary_embeddings(x, inv_freq):
    # x : [B, T, n_head, head_dim]
    seq_len = x.size(1)

    seq_inv_freq = inv_freq[:seq_len].unsqueeze(1).unsqueeze(0)
    # [1, T, 1, dim]
    cos, sin = seq_inv_freq.unbind(-1)

    return cos * x + rotate_half(x) * sin


class MultiHeadAttentionRoPE(nn.Module):
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
        self.inv_freq = precompute_inverse_frequency_matrix(self.head_dim, max_length=max_length)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        Q = apply_rotary_embeddings(Q, self.inv_freq)
        K = apply_rotary_embeddings(K, self.inv_freq)

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