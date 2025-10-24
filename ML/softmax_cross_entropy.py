import numpy as np

def softmax_cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute softmax cross-entropy loss for a batch of examples in a numerically stable way.

    logits: shape (batch_size, num_classes)
    labels: shape (batch_size,), integer class indices
    Returns: shape (batch_size,), loss for each example
    """
    # Shift logits for numerical stability (per row)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    
    # Compute log-sum-exp
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1))
    
    # Pick logits corresponding to true labels
    correct_logits = shifted_logits[np.arange(logits.shape[0]), labels]
    
    # Cross-entropy loss: -log(softmax) = -correct_logit + log_sum_exp
    # (batch_size,)
    loss = -correct_logits + log_sum_exp

    return loss.mean()
