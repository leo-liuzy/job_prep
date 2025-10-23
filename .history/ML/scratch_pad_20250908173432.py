def top_p_sampling(logits):
    # sort logits
    softmax(logits, dim=-1)
