import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def vanilla_edit(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """
    Greedy generation from scratch (temperature=0).
    """
    device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.inference_mode()
def speculative_edit(model, tokenizer, prompt: str, draft: str, max_new_tokens: int = 256):
    """
    Speculative edits:
      - First feed the draft sequence (tokens of the original region).
      - Accept tokens as long as model's greedy prediction matches the draft.
      - Once the model disagrees, switch to normal greedy generation.
    """
    device = next(model.parameters()).device

    # Prompt tokens
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Draft continuation tokens (the "speculation")
    draft_ids = tokenizer(draft, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Concatenate prompt + draft
    input_ids = torch.cat([prompt_ids, draft_ids], dim=1)
    generated = input_ids[:, : prompt_ids.size(1)]  # start with just prompt

    # 1. Verify draft tokens in bulk
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]  # ignore last token (no next-token for it)
    greedy_preds = logits.argmax(dim=-1)  # (1, seq_len-1)

    # Compare model predictions against the draft
    draft_tokens = draft_ids[0]
    matched_prefix_len = 0
    for i, token in enumerate(draft_tokens):
        # model prediction for position prompt+i must equal draft token
        if greedy_preds[0, prompt_ids.size(1) - 1 + i].item() == token.item():
            generated = torch.cat([generated, token.view(1, 1).to(device)], dim=1)
            matched_prefix_len += 1
        else:
            break

    # 2. If mismatch occurs, continue greedy generation from there
    remaining_budget = max_new_tokens - (generated.size(1) - prompt_ids.size(1))
    for _ in range(remaining_budget):
        outputs = model(input_ids=generated)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
