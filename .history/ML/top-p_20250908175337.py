import torch


def top_p_sampling(logits, p=0.9):
    if p == 1.0:
        # No truncation, just sample from full distribution
        probs = torch.softmax(logits, dim=-1)
        return probs
    probs = torch.softmax(logits, dim=-1)  # [B, V]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs > p
    mask[:, 0] = False  # keep top token
    sorted_probs[mask] = 0.0

    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    filtered_probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
    next_tokens = torch.multinomial(filtered_probs, 1)
    return next_tokens


def top_k_sampling(logits, k=50):
    """
    Sample a token from logits using top-k sampling.

    Args:
        logits: torch.Tensor of shape [vocab_size], raw logits.
        k: int, keep top-k tokens.

    Returns:
        int: sampled token id
    """
    if k <= 0:
        # No truncation, just sample from full distribution
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1)
        return next_tokens

    # Get top-k logits and their indices
    topk_probs, topk_indices = torch.topk(logits, k, dim=-1)

    filtered_probs = torch.zeros_like(logits).scatter(1, topk_indices, topk_probs)
    next_tokens = torch.multinomial(filtered_probs, 1)
    return next_tokens


def generate_sequences_batch(model, tokenizer, prompt, max_new_tokens=50, p=0.9, num_return_sequences=3, device="cpu"):
    """
    Generate multiple sequences in parallel with top-p sampling,
    stopping early for EOS tokens.
    """
    model.eval().to(device)
    eos_token_id = tokenizer.eos_token_id

    # Broadcast prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat(num_return_sequences, 1)

    # Track finished sequences
    finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]  # [B, V]

        # If a sequence is finished, set its logits to -inf (except EOS)
        if finished.any():
            logits[finished, :] = -float("inf")
            logits[finished, eos_token_id] = 0.0

        filtered_probs = top_p_sampline(logits, p=p)
        # next_tokens = torch.multinomial(filtered_probs, 1)  # [B, 1]

        input_ids = torch.cat([input_ids, next_tokens], dim=1)

        # Update finished mask
        finished |= next_tokens.squeeze(-1) == eos_token_id

        # If all sequences are finished, stop early
        if finished.all():
            break

    outputs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
    return outputs
