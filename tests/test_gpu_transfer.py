"""
Test GPU transfer and operations on loaded latents.
This tests what happens when latents are moved to GPU and used in training operations.
"""
import torch
import os
import sys
import tempfile
from safetensors.torch import save_file

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_gpu_transfer_of_loaded_latents():
    """Test moving loaded latents to GPU and using them in operations."""
    # Create test cache file
    test_latents = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        cache_path = f.name
    
    try:
        save_file({"latents_1x64x64_bfloat16": test_latents.cpu()}, cache_path)
        
        # Load as dataset loader does
        cpu_device = torch.device("cpu")
        with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as f:
            latents = f.get_tensor("latents_1x64x64_bfloat16", device=cpu_device, dtype=torch.float32)
        
        print(f"Loaded latents on CPU: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # Verify non-zero on CPU
        max_val_cpu = latents.abs().max().item()
        assert max_val_cpu > 1e-6, f"Latents should be non-zero on CPU, max={max_val_cpu}"
        
        # Move to GPU (if available)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            latents_gpu = latents.to(device)
            
            print(f"\nMoved to GPU: shape={latents_gpu.shape}, dtype={latents_gpu.dtype}")
            print(f"  min={latents_gpu.min().item():.6f}, max={latents_gpu.max().item():.6f}, mean={latents_gpu.mean().item():.6f}")
            
            # CRITICAL: Verify still non-zero on GPU
            max_val_gpu = latents_gpu.abs().max().item()
            assert max_val_gpu > 1e-6, f"CRITICAL: Latents became zeros on GPU! max={max_val_gpu}"
            
            # Test torch.randn_like
            noise = torch.randn_like(latents_gpu)
            print(f"\ntorch.randn_like result: shape={noise.shape}, dtype={noise.dtype}")
            print(f"  min={noise.min().item():.6f}, max={noise.max().item():.6f}, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
            
            # CRITICAL: Noise should never be all zeros
            max_noise = noise.abs().max().item()
            assert max_noise > 1e-6, f"CRITICAL: torch.randn_like produced zeros! max={max_noise}"
            
            # Test operations
            target = noise - latents_gpu
            print(f"\nTarget (noise - latents): shape={target.shape}, dtype={target.dtype}")
            print(f"  min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}")
            
            max_target = target.abs().max().item()
            assert max_target > 1e-6, f"CRITICAL: Target is all zeros! max={max_target}"
            
            print("✓ GPU transfer and operations test passed")
        else:
            print("Skipping GPU tests (CUDA not available)")
            
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)


def test_extreme_values_handling():
    """Test what happens with extreme values (like in real cache file)."""
    # Create tensor with extreme values (simulating the real cache file)
    # Use values that are very large but not Inf
    extreme_latents = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16) * 1e10
    
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        cache_path = f.name
    
    try:
        save_file({"latents_1x64x64_bfloat16": extreme_latents.cpu()}, cache_path)
        
        # Load
        cpu_device = torch.device("cpu")
        with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as f:
            latents = f.get_tensor("latents_1x64x64_bfloat16", device=cpu_device, dtype=torch.float32)
        
        print(f"Extreme values test - loaded: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # Check if they're Inf
        if torch.isinf(latents).any():
            print("WARNING: Extreme values became Inf")
            # Clamp them
            latents = torch.clamp(latents, min=-10.0, max=10.0)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)
            print(f"  After clamping: min={latents.min().item():.6f}, max={latents.max().item():.6f}")
        
        # Verify non-zero
        max_val = latents.abs().max().item()
        assert max_val > 1e-6, f"Latents should be non-zero, max={max_val}"
        
        print("✓ Extreme values handling test passed")
        
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)


if __name__ == "__main__":
    print("=" * 80)
    print("GPU TRANSFER TESTS")
    print("=" * 80)
    
    try:
        print("\n1. Testing GPU transfer of loaded latents...")
        test_gpu_transfer_of_loaded_latents()
        
        print("\n2. Testing extreme values handling...")
        test_extreme_values_handling()
        
        print("\n" + "=" * 80)
        print("ALL GPU TRANSFER TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




