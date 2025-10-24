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

    def forward(self, x, kv_cache):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape into heads: [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        prevK, prevV = kv_cache
        K = torch.concat([prevK, K], dim=-2)
        V = torch.concat([prevV, V], dim=-2)
        kv_cache_new = (K.detach(), V.detach())

        prev_seq_len = prevK.size(1)
        seq_len = prev_seq_len + T
        # [B, n_head, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        mask = self.mask[:, :, prev_seq_len:seq_len, :seq_len].to(x.device)
        # (B, n_head, T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, head_dim)
        out = attn @ V
        # (B, T, model_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, kv_cache_new

class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, dropout=0.1):
        super().__init__()  # important!
        self.n_head = n_head
        self.head_dim = model_size // n_head
        mask = torch.tril(torch.ones(max_length, max_length))
        self.register_buffer("mask", mask)
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.o_proj = nn.Linear(model_size, model_size)
        self.dropout = nnDropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # ^^ [B, n_head, T, head_dim]

        mask = self.mask[:, :, :T, :T]
        attn = self.dropout(attn)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask == 0, -float("inf"))
        attn = torch.softmax(scores, -1)
        out = attn @ V
        # ^^ [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        out = self.dropout(out)
        return out


class MultiQueryAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, dropout=0.1):
        super().__init__()  # important!
        self.head_dim = model_size // n_head
        self.n_head = n_head
        mask = torch.tril(torch.ones(max_length, max_length))
        self.register_buffer("mask", mask)
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, self.head_dim)
        self.v_proj = nn.Linear(model_size, self.head_dim)
        self.o_proj = nn.Linear(model_size, model_size)
        self.drpoout = Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        # ^^ [B, n_head, T, head_dim]

        mask = self.mask[:, :, :T, :T]
        attn = self.dropout(attn)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask == 0, -float("inf"))
        attn = torch.softmax(scores, -1)
        out = attn @ V
        # ^^ [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        out = self.dropout(out)
        return out
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, model_size, max_length, n_head, n_group, dropout=0.1):
        super().__init__()  # important!
        self.head_dim = model_size // n_head
        self.n_head = n_head
        self.n_group = n_group
        self.heads_per_group = n_head // n_group
        mask = torch.tril(torch.ones(max_length, max_length))
        self.register_buffer("mask", mask)
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, self.n_group * self.head_dim)
        self.v_proj = nn.Linear(model_size, self.n_group * self.head_dim)
        self.o_proj = nn.Linear(model_size, model_size)
        self.drpoout = Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # [B, n_head, T, head_dim]
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # [B, n_group, T, head_dim]
        K = K.view(B, T, self.n_group, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_group, self.head_dim).transpose(1, 2)
        
        # [B, n_group, heads_per_group, T, head_dim]
        Q = Q.view(B, self.n_group, self.heads_per_group, T, self.head_dim)
        # [B, n_group, 1, T, head_dim]
        K = K.unsqueeze(2)
        V = V.unsqueeze(2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)
        # ...
        out = attn @ V
        # [B, n_group, heads_per_group, T, head_dim]
        out = out.view(B, self.n_head, T, self.head_dim).transpose(1, 2).contiguous()
        out = out.view(B, T, D)

def top_p_sampling(logits, p):
    probs = torch.softmax(logits, dim=-1)

    sorted_indices, sorted_probs = torch.argsort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > p
    mask[:, 0] = False
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    filtered_probs = torch.zeros_like(probs).scatter(dim=1, sorted_indices, sorted_probs)
    next_tokens = torch.multinomial(filtered_probs, 1)
    return next_tokens




def generate_sequences_batch(model, tokenizer, prompt, max_new_tokens=50, p=0.9, num_return_sequences=3, device="cpu"):
    model.eval()
    model = model.to(device)
    eos_token_id = tokenizer.eos_token_id
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat(num_return_sequences, 1)

    finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]

        if finished.any():
            logits[finished, :] = float("-inf")
            logits[finished, eos_token_id] = 0.0
        # [num_return_sequences, 1]
        next_tokens = top_p_sampling(logits, p=p)

        input_ids = torch.cat([input_ids, next_tokens], dim=-1)        

        finished |= next_tokens.squeeze(-1) == eos_token_id

        if finished.all():
            break
    outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return outputs
     