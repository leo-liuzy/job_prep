import torch


def precompute_frequency_complex_matrix(head_dim: int, seq_len, theta: float = 10000.0):
    # 1 / (10000 ^ (2i /d)); i = 0, ..., head_dim/2
    # (head_dim/2, )
    theta = 1 / (theta ** torch.arange(0, head_dim, 2).float() / head_dim)

    # (seq_len,)
    m = torch.arange(0, seq_len)

    # (seq_len, head_dim/2)
    frequency_matrix = torch.outer(m, theta)

    # (seq_len, head_dim/2)
    # [m_1*theta_1 m_1*theta_2 ... m_1*theta_head_dim/2]
    # ...
    # [m_m*theta_1 m_m*theta_2 ... m_m*theta_head_dim/2]
    freq_complex = torch.polar(torch.ones_like(frequency_matrix), frequency_matrix)

    return freq_complex


def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor):
    # (batch_size, seq_len, H, head_dim) -> (batch_size, seq_len, H, head_dim/2, 2) -> (batch_size, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))  # TODO: use reshape instead of view

    # (seq_len, head_dim/2) -> (1, seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

    # (batch_size, seq_len, H, head_dim/2) -> (batch_size, seq_len, H, head_dim/2)
    x_rotate = x_complex * freq_complex

    # (batch_size, seq_len, H, head_dim/2) -> (batch_size, seq_len, H, head_dim/2, 2)
    x_rotate = torch.view_as_real(x_rotate)

    # (batch_size, seq_len, H, head_dim/2, 2) -> (batch_size, seq_len, H, head_dim)
    x_rotate = x_rotate.reshape(*x.shape).type_as(x)

    return x_rotate
