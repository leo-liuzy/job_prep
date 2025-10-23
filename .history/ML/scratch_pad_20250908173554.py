def top_p_sampling(logits):
    # [B, T, V]
    # sort logits
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1)
