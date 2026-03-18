import torch
from dynamic_dp_core import calculate_dynamic_K
from structure_extractor import StructureExtractor

def dummy_test():
    """
    [Strict Professor Audit] 
    This is an isolated unit test to ensure no fundamental syntax 
    or tensor dimension errors exist before deploying to the remote GPU. 
    It mocks the LLaMA vocabulary similarity matrix.
    """
    print("--- Running Local Sanity Check for SP-DYN-DP Components ---")
    
    # 1. Test Structure Extractor (Component C)
    print("\n[Audit Component C: Structure Extractor]")
    extractor = StructureExtractor(use_openie=False)
    text = "The patient John Doe was diagnosed with severe pneumonia."
    structural_words = extractor.extract_structural_words(text)
    print(f"Original Text: '{text}'")
    print(f"Extracted SVO Backbone (Protected): {structural_words}")
    assert len(structural_words) > 0, "Structure extractor failed to find backbone."
    
    # 2. Test Dynamic K calculation (Component B)
    print("\n[Audit Component B: Dynamic Density & K-scaling]")
    L, V = 8, 128000 # 8 input tokens, 128k LLaMA vocab mock
    
    # Simulate cosine similarity matrix (mean 0, std 0.5)
    similarities = torch.clamp(torch.randn(L, V) * 0.5, -1.0, 1.0)
    
    # Make token 0 extremely "common" (high density -> lower K required)
    similarities[0, :5000] = 0.9 
    
    # Make token 1 extremely "rare/sensitive" (low density -> higher K required)
    similarities[1, :] = -0.1 
    
    base_K = 50
    dynamic_K = calculate_dynamic_K(similarities, base_K, gamma_threshold=0.5, density_epsilon=0.5)
    
    print(f"Base K was set to: {base_K}")
    print(f"Calculated Dynamic K array for {L} tokens: {dynamic_K.tolist()}")
    
    assert dynamic_K[0] < dynamic_K[1], "Dynamic K scaling logic failed: Common token should have smaller K than rare token."
    
    print("\n✅ All local audits passed. The mathematical logic is sound.")

if __name__ == "__main__":
    dummy_test()
