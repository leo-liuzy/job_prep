def top_p_sampling(logits, p):
    # [B, T, V]
    # sort logits
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > p
    mask[:, 0] = False  # keep at least 1 token
    sorted_probs[mask] = 0.0
