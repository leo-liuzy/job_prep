head_dim = 8
theta_base = 10000
seq_len = 10

# (seq_len, head_dim/2
# [m_1*theta_1 m_1*theta_2 ... m_1*theta_head_dim/2]
# ...


def precompute_inverse_frequency_matrix(dim, theta_base, max_seq_len: int = 4096) -> None:
    theta = 1.0 / (theta_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    seq_idx = torch.arange(max_seq_len, dtype=theta.dtype, device=theta.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()

    # cache includes both the cos and sin components and so the output shape is
    # [max_seq_len, dim // 2, 2]
    inv_freq = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
    inv_freq = torch.repeat_interleave(inv_freq, repeats=2, dim=-2)
    inv_freq = inv_freq
    return inv_freq


def rotate_half(x):
    x_shape = x.shape
    even_x = x[..., ::2]
    odd_x = x[..., 1::2]
    x = torch.stack([-odd_x, even_x], dim=-1).view(*x_shape[:-1], -1)
    return x


def apply_rotary_embeddings(x, inv_freq):
    # x : B, T, n_head, head_dim
    seq_len = x.size(1)

    seq_inv_freq = inv_freq[:seq_len].unsqueeze(1).unsqueeze(0)
    cos, sin = seq_inv_freq.unbind(-1)

    return cos * x + rotate_half(x) * sin
