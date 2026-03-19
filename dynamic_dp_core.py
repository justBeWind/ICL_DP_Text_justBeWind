import torch

def calculate_dynamic_K(similarities, base_K, top_n_density=1000, density_epsilon=0.1):
    """
    Computes dynamic K for each token based on semantic density directly from the similarity matrix.
    
    [Strict Professor Notice - Model Agnostic Update]: 
    Fixed thresholds (like gamma=0.2) fail across different models (LLaMA vs Qwen) because high-dimensional 
    embeddings (e.g., 4096D) are naturally orthogonal, and their absolute cosine similarities vary wildly.
    Instead of counting neighbors above a hardcoded threshold, we compute the sum of the Top-N similarities 
    for each token. We then apply **Sequence-Relative Min-Max Normalization**. 
    
    This guarantees that within ANY given sentence, the token with the densest neighborhood gets K=1, 
    the most isolated gets K=base_K, and the rest scale proportionally, completely regardless of the 
    underlying LLM!
    
    Args:
        similarities: Tensor of shape (L, V) containing cosine similarities between L input tokens and V vocab tokens.
        base_K: The maximum/base K value.
        top_n_density: Number of top similarity scores to sum up to define "local density".
        density_epsilon: Privacy budget allocated specifically for perturbing the density metric itself.
        
    Returns:
        Tensor of shape (L,) containing dynamic K values for each token.
    """
    # 1. Density Calculation (f_x) - Model Agnostic
    # Take the top N most similar tokens from the vocab to represent local manifold density
    top_sims, _ = similarities.topk(top_n_density, dim=1)
    raw_density = top_sims.sum(dim=1).float() # Shape: (L,)
    
    # Add Laplace noise to density counts to satisfy DP for density itself
    # Sensitivity of sum of top_n cosine similarities (where max sim is 1) is roughly top_n.
    scale = top_n_density / density_epsilon if density_epsilon > 0 else 0.0
    if scale > 0:
        density_noise = torch.distributions.Laplace(0, scale).sample(raw_density.shape).to(similarities.device)
        noisy_density = raw_density + density_noise
    else:
        noisy_density = raw_density
    
    # 2. Sequence-Relative Min-Max Normalization
    min_val = noisy_density.min()
    max_val = noisy_density.max()
    
    # Prevent division by zero if all densities are identical
    if max_val - min_val < 1e-6:
        normalized_density = torch.zeros_like(noisy_density)
    else:
        normalized_density = (noisy_density - min_val) / (max_val - min_val)
    
    # 3. Dynamic K scaling
    # Low density -> rare sensitive word -> normalized = 0.0 -> K = base_K
    # High density -> common word -> normalized = 1.0 -> K = 1
    dynamic_K = torch.floor(base_K * (1.0 - normalized_density)) + 1
    dynamic_K = torch.clamp(dynamic_K, min=1, max=base_K).long()
    
    return dynamic_K
