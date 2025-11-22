"""
Test batch creation process - this is where zeros might be introduced.
Tests the exact process used in BucketBatchManager.__getitem__
"""
import torch
import os
import sys
import tempfile
from safetensors.torch import save_file

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_batch_creation_process():
    """Test the exact batch creation process from BucketBatchManager."""
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
        
        # Simulate EXACT process from BucketBatchManager.__getitem__
        batch_tensor_data = {}
        varlen_keys = set()
        
        # Load as dataset loader does
        cpu_device = torch.device("cpu")
        sd_latent = {}
        with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as f:
            for key in f.keys():
                tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
                if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                    tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                sd_latent[key] = tensor
        
        # Process keys (simulating the key processing logic)
        for key in sd_latent.keys():
            is_varlen_key = key.startswith("varlen_")
            content_key = key
            
            if is_varlen_key:
                content_key = content_key.replace("varlen_", "")
            
            if content_key.endswith("_mask"):
                pass
            else:
                content_key = content_key.rsplit("_", 1)[0]  # remove dtype
                if content_key.startswith("latents_"):
                    content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW
            
            if content_key not in batch_tensor_data:
                batch_tensor_data[content_key] = []
            
            batch_tensor_data[content_key].append(sd_latent[key])
            
            if is_varlen_key:
                varlen_keys.add(content_key)
        
        # Stack tensors (this is where issues might occur)
        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])
        
        # Verify latents
        if "latents" in batch_tensor_data:
            latents = batch_tensor_data["latents"]
            
            print(f"Batch creation test - latents: shape={latents.shape}, dtype={latents.dtype}")
            print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
            
            # CRITICAL: Must not be zeros after stacking
            max_val = latents.abs().max().item()
            assert max_val > 1e-6, f"CRITICAL: Latents became zeros after batch creation! max={max_val}"
            assert latents.dtype == torch.float32, f"Should be float32, got {latents.dtype}"
            
            print("✓ Batch creation test passed")
        else:
            raise AssertionError("Expected 'latents' key not found in batch_tensor_data")
            
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)


def test_real_cache_batch_creation():
    """Test batch creation with real cache file."""
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    if not os.path.exists(cache_file):
        print(f"WARNING: Cache file not found: {cache_file}")
        return
    
    print(f"Testing real cache file batch creation: {cache_file}")
    
    # Load as dataset loader does
    cpu_device = torch.device("cpu")
    sd_latent = {}
    
    with MemoryEfficientSafeOpen(cache_file, disable_numpy_memmap=False) as f:
        for key in f.keys():
            tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
            if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            sd_latent[key] = tensor
    
    # Process and stack
    batch_tensor_data = {}
    for key in sd_latent.keys():
        content_key = key.rsplit("_", 1)[0]  # remove dtype
        if content_key.startswith("latents_"):
            content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW
        
        if content_key not in batch_tensor_data:
            batch_tensor_data[content_key] = []
        batch_tensor_data[content_key].append(sd_latent[key])
    
    for key in batch_tensor_data.keys():
        batch_tensor_data[key] = torch.stack(batch_tensor_data[key])
    
    # Verify
    if "latents" in batch_tensor_data:
        latents = batch_tensor_data["latents"]
        
        print(f"Real cache batch creation - latents: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        max_val = latents.abs().max().item()
        if max_val < 1e-6:
            raise AssertionError(f"CRITICAL: Real cache produces zeros after batch creation! max={max_val}")
        
        print(f"✓ Real cache batch creation test passed: max={max_val:.6f}")
    else:
        print("WARNING: 'latents' key not found")


if __name__ == "__main__":
    print("=" * 80)
    print("BATCH CREATION TESTS")
    print("=" * 80)
    
    try:
        print("\n1. Testing batch creation with test cache...")
        test_batch_creation_process()
        
        print("\n2. Testing batch creation with real cache...")
        test_real_cache_batch_creation()
        
        print("\n" + "=" * 80)
        print("ALL BATCH CREATION TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




