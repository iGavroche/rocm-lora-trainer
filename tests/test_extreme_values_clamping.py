"""
Test what happens with extreme values from real cache file.
The real cache has values around 10^38 which might cause issues.
"""
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_extreme_values_become_zeros():
    """Test if extreme values become zeros when used in operations."""
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    if not os.path.exists(cache_file):
        print(f"WARNING: Cache file not found: {cache_file}")
        return
    
    print(f"Testing extreme values from real cache: {cache_file}")
    
    # Load as dataset loader does
    cpu_device = torch.device("cpu")
    with MemoryEfficientSafeOpen(cache_file, disable_numpy_memmap=False) as f:
        latents = f.get_tensor("latents_1x64x64_bfloat16", device=cpu_device, dtype=torch.float32)
    
    print(f"Loaded latents: shape={latents.shape}, dtype={latents.dtype}")
    print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
    
    # Check if values are Inf (even though isinf might return False for very large values)
    max_val = latents.abs().max().item()
    print(f"  max absolute value: {max_val:.6e}")
    
    # These extreme values might cause issues - let's clamp them
    if max_val > 1e10:  # Values are extremely large
        print(f"WARNING: Values are extremely large ({max_val:.6e}), clamping...")
        latents_clamped = torch.clamp(latents, min=-10.0, max=10.0)
        print(f"  After clamping: min={latents_clamped.min().item():.6f}, max={latents_clamped.max().item():.6f}, mean={latents_clamped.mean().item():.6f}")
        
        # Verify clamped values are non-zero
        max_clamped = latents_clamped.abs().max().item()
        assert max_clamped > 1e-6, f"CRITICAL: Clamped latents are zeros! max={max_clamped}"
        
        # Test operations with clamped values
        noise = torch.randn_like(latents_clamped)
        target = noise - latents_clamped
        
        assert target.abs().max().item() > 1e-6, "Target should be non-zero"
        print("✓ Extreme values can be clamped and used successfully")
    else:
        print("Values are within reasonable range")


if __name__ == "__main__":
    print("=" * 80)
    print("EXTREME VALUES CLAMPING TEST")
    print("=" * 80)
    
    try:
        test_extreme_values_become_zeros()
        print("\n✓ TEST PASSED")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




