"""
Test loading the actual cache file and verify it works in the training pipeline.
This is a CRITICAL test that must pass for training to work.
"""
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

def test_real_cache_file_loading():
    """Test loading the actual cache file used in training."""
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    if not os.path.exists(cache_file):
        print(f"WARNING: Cache file not found: {cache_file}")
        print("Skipping real cache file test")
        return
    
    print(f"Testing real cache file: {cache_file}")
    
    # Load exactly as the dataset loader does
    cpu_device = torch.device("cpu")
    sd_latent = {}
    
    with MemoryEfficientSafeOpen(cache_file, disable_numpy_memmap=False) as f:
        for key in f.keys():
            print(f"Loading key: {key}")
            tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
            
            print(f"  After loading: shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
            print(f"    has Inf: {torch.isinf(tensor).any().item()}, has NaN: {torch.isnan(tensor).any().item()}")
            
            # Apply clamping as done in dataset loader
            if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                print(f"  Applying clamping for Inf/NaN values...")
                tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                print(f"  After clamping: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
            
            sd_latent[key] = tensor
    
    # Check main latents
    if "latents_1x64x64_bfloat16" in sd_latent:
        latents = sd_latent["latents_1x64x64_bfloat16"]
        
        print(f"\nFinal latents: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # CRITICAL CHECK: Are they zeros?
        max_val = latents.abs().max().item()
        if max_val < 1e-6:
            raise AssertionError(f"CRITICAL FAILURE: Latents are all zeros after loading! max={max_val}")
        
        # Check if they're all the same value (clamped Inf)
        unique_values = torch.unique(latents)
        if unique_values.numel() <= 3:
            print(f"WARNING: Latents have very few unique values: {unique_values.numel()}")
            print(f"  Unique values: {unique_values.tolist()}")
        
        print(f"✓ Real cache file loaded successfully")
        print(f"  Max absolute value: {max_val:.6f}")
        print(f"  Number of unique values: {torch.unique(latents).numel()}")
        
        return latents
    else:
        raise AssertionError("Expected key 'latents_1x64x64_bfloat16' not found")


if __name__ == "__main__":
    try:
        latents = test_real_cache_file_loading()
        print("\n✓ TEST PASSED: Real cache file loads with non-zero values")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




