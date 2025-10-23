import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F


def sample_next_token(logits, temperature=1.0, top_p=0.9):
    """Sample from logits with nucleus (top-p) sampling."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # sort probs descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # keep only tokens within cumulative prob <= top_p
    cutoff = cum_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum()

    next_id = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[next_id].item(), probs


def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens=50,
    k_draft=5,
    temperature=1.0,
    top_p=0.9,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device).eval()
    draft_model.to(device).eval()

    enc = tokenizer(prompt, return_tensors="pt")
    cur_ids = enc["input_ids"].to(device)  # (1, seq_len)
    generated = []

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            # forward through draft
            draft_out = draft_model(cur_ids)
            draft_logits = draft_out.logits[0, -1]
            draft_id, draft_probs = sample_next_token(draft_logits, temperature=temperature, top_p=top_p)

            # forward through target
            target_out = target_model(cur_ids)
            target_logits = target_out.logits[0, -1]
            target_probs = F.softmax(target_logits, dim=-1)

            # acceptance probability
            p_d = draft_probs[draft_id].item()
            p_t = target_probs[draft_id].item()
            accept_prob = min(1.0, p_t / (p_d + 1e-9))

            if torch.rand(1).item() < accept_prob:
                next_id = draft_id  # accept draft token
            else:
                # sample directly from target distribution
                next_id, _ = sample_next_token(target_logits, temperature=temperature, top_p=top_p)

            generated.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]], device=device)], dim=1)

            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True), generated


if __name__ == "__main__":
    draft_name = "distilgpt2"
    target_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(draft_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft = AutoModelForCausalLM.from_pretrained(draft_name)
    target = AutoModelForCausalLM.from_pretrained(target_name)

    prompt = "The scientist carefully mixed the chemicals and"
    out_text, _ = speculative_decode(target, draft, tokenizer, prompt, max_new_tokens=40, k_draft=5, temperature=0.8)

    print("PROMPT:", prompt)
    print("GENERATED:", out_text)
