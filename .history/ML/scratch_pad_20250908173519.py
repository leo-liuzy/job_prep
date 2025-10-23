def top_p_sampling(logits):
    # sort logits
    probs = softmax(logits, dim=-1)
