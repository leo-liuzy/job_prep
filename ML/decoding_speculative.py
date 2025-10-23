import torch
import torch.nn.functional as F

def speculative_decode(draft_model, target_model, prompt_ids, max_tokens=50, k=4, temperature=1.0):
    """
    Minimal speculative decoding implementation.
    
    Speculative decoding speeds up inference by using a small "draft" model to generate
    candidate tokens, then verifying them in parallel with a large "target" model.
    This maintains the same output distribution as standard sampling from the target model.
    
    Args:
        draft_model: Small, fast model for speculation (generates guesses)
        target_model: Large, accurate model for verification (checks guesses)
        prompt_ids: Input token IDs (tensor of shape [batch_size, seq_len])
        max_tokens: Maximum number of tokens to generate
        k: Number of tokens to speculate ahead (typically 4-8)
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated token IDs concatenated with prompt
    """
    tokens = prompt_ids.clone()
    
    # Generate tokens in chunks of size k
    for _ in range(max_tokens // k):
        
        # ===== DRAFT PHASE: Fast Model Generates k Speculative Tokens =====
        # The draft model quickly generates k candidate tokens autoregressively
        # This is fast because the draft model is small
        draft_tokens = []  # Store the k generated tokens
        draft_probs_list = []  # Store probability distributions for each token
        current = tokens  # Start with all tokens generated so far
        
        for _ in range(k):
            with torch.no_grad():
                # Get logits from draft model for next token
                logits = draft_model(current)[:, -1, :]  # [batch, vocab_size]
                
                # Convert to probabilities
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Sample next token from draft distribution
                next_token = torch.multinomial(probs, 1)
                
                # Save both the token and its probability distribution
                draft_tokens.append(next_token)
                draft_probs_list.append(probs)
                
                # Add token to sequence for next iteration
                current = torch.cat([current, next_token], dim=1)
        
        # ===== VERIFICATION PHASE: Target Model Verifies All k Tokens in Parallel =====
        # KEY INSIGHT: Instead of generating k tokens one-by-one with the target model
        # (which would take k forward passes), we verify all k draft tokens at once
        # (which takes only 1 forward pass). This is where the speedup comes from!
        
        with torch.no_grad():
            # Run target model once on the entire sequence including all k draft tokens
            target_logits = target_model(current)
            
            # Get target probabilities for the k positions where we generated draft tokens
            # Indexing explanation: logits[:, i, :] predicts the token at position i+1
            # If current = [original_tokens, draft_1, draft_2, ..., draft_k]
            # Then logits[:, -k-1, :] predicts draft_1 (the token at position -k)
            #      logits[:, -k, :]   predicts draft_2 (the token at position -k+1)
            #      ...
            #      logits[:, -2, :]   predicts draft_k (the token at position -1)
            # So [:, -k-1:-1, :] gives us k logit vectors for predicting our k draft tokens
            target_probs = F.softmax(target_logits[:, -k-1:-1, :] / temperature, dim=-1)
        
        # ===== ACCEPTANCE/REJECTION: Decide Which Draft Tokens to Keep =====
        # We use a probabilistic acceptance criterion that ensures the final distribution
        # matches what we would get if we had sampled from the target model directly
        
        accepted = 0
        for i in range(k):
            draft_token = draft_tokens[i]
            
            # Get the probability that draft model assigned to the token it generated
            draft_prob = draft_probs_list[i][0, draft_token]
            
            # Get the probability that target model assigns to that same token
            target_prob = target_probs[0, i, draft_token]
            
            # Accept token with probability min(1, target_prob / draft_prob)
            # - If target prob > draft prob: always accept (target likes it more)
            # - If target prob < draft prob: accept probabilistically
            # This ensures we don't over-represent tokens that draft model favors
            accept_prob = torch.min(torch.ones(1), target_prob / draft_prob)
            
            if torch.rand(1) < accept_prob:
                # Token is accepted! Add it to our generated sequence
                tokens = torch.cat([tokens, draft_token], dim=1)
                accepted += 1
            else:
                # ===== REJECTION SAMPLING: Token Rejected, Resample =====
                # When we reject a token, we can't just pick the next one randomly
                # We need to sample from an "adjusted" distribution that accounts for
                # the fact that we already rejected some probability mass
                
                # Adjusted distribution = max(0, target_probs - draft_probs)
                # This removes the probability mass we already "used up" in the rejection
                adjusted_probs = torch.clamp(
                    target_probs[0, i] - draft_probs_list[i][0], 
                    min=0
                )
                
                # Renormalize to make it a valid probability distribution
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                # Sample a corrected token from the adjusted distribution
                corrected_token = torch.multinomial(adjusted_probs, 1).unsqueeze(0)
                tokens = torch.cat([tokens, corrected_token], dim=1)
                
                # Stop verifying further tokens - they were based on a rejected token
                break
        
        # ===== BONUS TOKEN: If All k Tokens Accepted, Generate One More =====
        # If we accepted all k speculative tokens, we can generate one additional
        # token from the target model "for free" since we already did a forward pass
        if accepted == k:
            probs = F.softmax(target_logits[:, -1, :] / temperature, dim=-1)
            bonus_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, bonus_token], dim=1)
        
        # Stop if we've generated enough tokens
        if tokens.shape[1] >= len(prompt_ids) + max_tokens:
            break
    
    return tokens


# ===== Example Usage with Mock Models =====

class MockModel(torch.nn.Module):
    """
    Simple mock model for demonstration.
    In practice, you'd use real language models like:
    - Draft: Small model (e.g., 125M params)
    - Target: Large model (e.g., 7B params)
    """
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size, vocab_size)
    
    def forward(self, x):
        # Simple embedding lookup simulation
        one_hot = F.one_hot(x, num_classes=1000).float()
        return self.linear(one_hot.sum(dim=1, keepdim=True))

# Initialize models
draft = MockModel()  # Fast, small model
target = MockModel()  # Slow, large model

# Generate tokens
prompt = torch.tensor([[1, 2, 3, 4]])
result = speculative_decode(draft, target, prompt, max_tokens=20, k=4)

print(f"Input length: {prompt.shape[1]}")
print(f"Generated {result.shape[1] - prompt.shape[1]} new tokens")
print(f"Output shape: {result.shape}")
print(f"\nSpeedup: Instead of ~20 target model calls, we made ~{(result.shape[1] - prompt.shape[1]) // 4} calls")