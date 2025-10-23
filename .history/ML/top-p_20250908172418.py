import torch


def top_p_filtering_batch(logits, p=0.9):
    """
    Batched top-p (nucleus) filtering for logits of shape [batch_size, vocab_size].
    Returns probabilities after filtering/renormalizing.
    """
    probs = torch.softmax(logits, dim=-1)  # [B, V]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens outside the nucleus
    mask = cumulative_probs > p
    mask[:, 0] = False  # always keep at least the top token
    sorted_probs[mask] = 0.0

    # Renormalize
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Scatter back into original order
    filtered_probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
    return filtered_probs


def generate_sequences_batch(model, tokenizer, prompt, max_new_tokens=50, p=0.9, num_return_sequences=3, device="cpu"):
    """
    Generate multiple sequences in parallel using top-p sampling.

    Args:
        model: causal LM (e.g. GPT2LMHeadModel)
        tokenizer: matching tokenizer
        prompt: str, input text
        max_new_tokens: int, number of tokens to generate
        p: float, nucleus probability threshold
        num_return_sequences: int, number of samples to return
        device: str
    """
    model.eval().to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat(num_return_sequences, 1)  # broadcast prompt

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]  # [B, V]

        # Apply batched nucleus filtering
        filtered_probs = top_p_filtering_batch(logits, p=p)

        # Sample next tokens for each sequence
        next_tokens = torch.multinomial(filtered_probs, num_samples=1)  # [B, 1]

        # Append to all sequences
        input_ids = torch.cat([input_ids, next_tokens], dim=1)

    # Decode each sequence
    outputs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
    return outputs
