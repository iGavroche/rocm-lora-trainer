"""
Test that DataLoader workers correctly load latents.
This simulates the actual DataLoader worker process.
"""
import torch
import os
import sys
import tempfile
from safetensors.torch import save_file

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def simulate_dataloader_worker_load(cache_path):
    """Simulate exactly what a DataLoader worker does when loading."""
    # This is the exact code from BucketBatchManager.__getitem__
    cpu_device = torch.device("cpu")
    sd_latent = {}
    
    with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as f:
        for key in f.keys():
            # Load as float32 directly to avoid ROCm bfloat16->float32 conversion bug
            tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
            # Clamp Inf/NaN values if present
            if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            sd_latent[key] = tensor
    
    return sd_latent


def test_dataloader_worker_simulation():
    """Test that DataLoader worker simulation produces non-zero latents."""
    # Create test cache file
    test_latents = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    test_image_latents = torch.randn(20, 1, 64, 64, dtype=torch.bfloat16)
    
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        cache_path = f.name
    
    try:
        save_file({
            "latents_1x64x64_bfloat16": test_latents.cpu(),
            "latents_image_1x64x64_bfloat16": test_image_latents.cpu()
        }, cache_path)
        
        # Simulate DataLoader worker
        sd_latent = simulate_dataloader_worker_load(cache_path)
        
        # Verify
        assert "latents_1x64x64_bfloat16" in sd_latent
        latents = sd_latent["latents_1x64x64_bfloat16"]
        
        print(f"DataLoader worker simulation - latents: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # CRITICAL: Must not be zeros
        max_val = latents.abs().max().item()
        assert max_val > 1e-6, f"CRITICAL: DataLoader worker produced zeros! max={max_val}"
        assert latents.dtype == torch.float32, f"Should be float32, got {latents.dtype}"
        
        print("✓ DataLoader worker simulation passed")
        
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)


def test_real_cache_with_dataloader_simulation():
    """Test loading real cache file using DataLoader worker simulation."""
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    if not os.path.exists(cache_file):
        print(f"WARNING: Cache file not found: {cache_file}")
        return
    
    print(f"Testing real cache file with DataLoader worker simulation: {cache_file}")
    
    # Simulate DataLoader worker
    sd_latent = simulate_dataloader_worker_load(cache_file)
    
    if "latents_1x64x64_bfloat16" in sd_latent:
        latents = sd_latent["latents_1x64x64_bfloat16"]
        
        print(f"Loaded latents: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # CRITICAL CHECK
        max_val = latents.abs().max().item()
        if max_val < 1e-6:
            raise AssertionError(f"CRITICAL: Real cache file produces zeros in DataLoader worker! max={max_val}")
        
        # Check if values are reasonable (not all Inf)
        if torch.isinf(latents).any():
            print(f"WARNING: Latents contain Inf values")
        if torch.isnan(latents).any():
            print(f"WARNING: Latents contain NaN values")
        
        # Check if they're all the same (would indicate clamping issue)
        unique_count = torch.unique(latents).numel()
        if unique_count < 10:
            print(f"WARNING: Very few unique values: {unique_count}")
        
        print(f"✓ Real cache file works in DataLoader worker simulation")
        print(f"  Max value: {max_val:.6f}")
        print(f"  Unique values: {unique_count}")
        
    else:
        raise AssertionError("Expected key not found")


if __name__ == "__main__":
    print("=" * 80)
    print("DATALOADER WORKER TESTS")
    print("=" * 80)
    
    try:
        print("\n1. Testing DataLoader worker simulation with test cache...")
        test_dataloader_worker_simulation()
        
        print("\n2. Testing DataLoader worker simulation with real cache...")
        test_real_cache_with_dataloader_simulation()
        
        print("\n" + "=" * 80)
        print("ALL DATALOADER WORKER TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




