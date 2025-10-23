import torch


head_dim = 8
theta_base = 10000
seq_len = 10

# (seq_len, head_dim/2
# [m_1*theta_1 m_1*theta_2 ... m_1*theta_head_dim/2]
# ...


def precompute_linear_bias_matrix(max_seq_len: int = 4096):
    # 1 / (10000 ^ (2i /d)); i = 0, ..., head_dim/2
    i_idx = torch.arange(max_seq_len).unsqueeze(0)

    j_idx = torch.arange(max_seq_len).unsqueeze(1)

    mat = i_idx - j_idx

    return 

class MultiHeadAttentionAlibi(nn.Module):
    def __init__(self, model_size, max_length, n_head, m, dropout=0.1):
        super().__init__()  # important!
        assert model_size % n_head == 0, "model_size must be divisible by n_head"
        self.model_size = model_size
        self.n_head = n_head
        self.head_dim = model_size // n_head
        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))
        # Difference
        self.register_buffer("linear_bias", torch.tril(precompute_linear_bias_matrix(max_length, max_length)))
        # Difference
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)
        self.linear_bias = self.linear_bias.unsqueeze(0).unsqueeze(0)
        self.m = m

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

        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        # Difference
        linear_bias = self.linear_bias[:, :, :T, :T].to(x.device)
        scores += self.m * linear_bias
        # Difference

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