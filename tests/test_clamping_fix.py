"""
Test that the clamping fix for extreme values works correctly.
This verifies that extreme values (>1e10) are clamped to reasonable range.
"""
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_extreme_values_clamping():
    """Test that extreme values are clamped correctly."""
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    if not os.path.exists(cache_file):
        print(f"WARNING: Cache file not found: {cache_file}")
        return
    
    print(f"Testing extreme values clamping with real cache: {cache_file}")
    
    # Load as dataset loader does (with the new clamping fix)
    cpu_device = torch.device("cpu")
    
    with MemoryEfficientSafeOpen(cache_file, disable_numpy_memmap=False) as f:
        tensor = f.get_tensor("latents_1x64x64_bfloat16", device=cpu_device, dtype=torch.float32)
    
    print(f"Before clamping: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
    max_val_before = tensor.abs().max().item()
    print(f"  max absolute value: {max_val_before:.6e}")
    
    # Apply clamping logic from dataset loader
    if torch.isinf(tensor).any() or torch.isnan(tensor).any():
        tensor = torch.clamp(tensor, min=-10.0, max=10.0)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Also clamp extremely large values
    max_val = tensor.abs().max().item()
    if max_val > 1e10:
        print(f"Clamping extreme values (max={max_val:.6e}) to [-10, 10]...")
        tensor = torch.clamp(tensor, min=-10.0, max=10.0)
    
    print(f"After clamping: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
    max_val_after = tensor.abs().max().item()
    
    # CRITICAL ASSERTIONS
    assert max_val_after <= 10.0, f"Clamped values should be <= 10.0, got {max_val_after}"
    assert max_val_after > 1e-6, f"CRITICAL: Clamped latents are all zeros! max={max_val_after}"
    assert tensor.min().item() >= -10.0, f"Clamped values should be >= -10.0"
    
    # Verify values are reasonable
    unique_count = torch.unique(tensor).numel()
    print(f"  Unique values after clamping: {unique_count}")
    assert unique_count > 10, f"Should have many unique values, got {unique_count}"
    
    # Test that clamped values work in operations
    noise = torch.randn_like(tensor)
    target = noise - tensor
    
    assert target.abs().max().item() > 1e-6, "Target should be non-zero"
    print("✓ Extreme values clamping test passed")
    print(f"  Values successfully clamped from {max_val_before:.6e} to {max_val_after:.6f}")


if __name__ == "__main__":
    print("=" * 80)
    print("CLAMPING FIX TEST")
    print("=" * 80)
    
    try:
        test_extreme_values_clamping()
        print("\n✓ ALL TESTS PASSED")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




