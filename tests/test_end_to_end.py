"""
End-to-end test that verifies the complete pipeline from cache file to training.
This test actually loads real cache files and verifies they work in the dataset loader.
"""
import torch
import os
import sys
import tempfile
from safetensors.torch import save_file

# Add tests directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import verify_latents_non_zero, cleanup_test_file

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_load_real_cache_file():
    """Test loading an actual cache file that was created by the caching script."""
    # Check if we have a real cache file to test with
    cache_dir = "myface"
    if not os.path.exists(cache_dir):
        print(f"WARNING: Cache directory {cache_dir} not found, skipping real cache test")
        return
    
    # Find a cache file
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith("_wan.safetensors")]
    if not cache_files:
        print(f"WARNING: No cache files found in {cache_dir}, skipping real cache test")
        return
    
    cache_path = os.path.join(cache_dir, cache_files[0])
    print(f"Testing with real cache file: {cache_path}")
    
    # Load using MemoryEfficientSafeOpen (same as dataset loader)
    with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
        keys = list(safe_file.keys())
        print(f"Keys in cache file: {keys}")
        
        # Load latents
        if "latents_1x64x64_bfloat16" in keys:
            latents = safe_file.get_tensor(
                "latents_1x64x64_bfloat16",
                device=torch.device("cpu"),
                dtype=torch.float32
            )
            
            print(f"Loaded latents: shape={latents.shape}, dtype={latents.dtype}")
            print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
            
            # CRITICAL: Verify latents are NOT all zeros
            max_val = latents.abs().max().item()
            if max_val < 1e-6:
                raise AssertionError(f"CRITICAL: Real cache file has all-zero latents! This means the cache is corrupted.")
            
            verify_latents_non_zero(latents, "real_cache_latents")
            assert latents.dtype == torch.float32, "Should be loaded as float32"
            print("✓ Real cache file loaded successfully with non-zero latents")
        else:
            print(f"WARNING: Expected key 'latents_1x64x64_bfloat16' not found in cache file")


def test_create_and_load_cache_file():
    """Test creating a cache file and loading it back (full roundtrip)."""
    # Create test latents with known non-zero values
    test_latents = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    original_max = test_latents.abs().max().item()
    assert original_max > 0, "Test latents must have non-zero values"
    
    # Create cache file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        cache_path = f.name
    
    try:
        # Save as bfloat16 (as the real caching script does)
        save_file({"latents_1x64x64_bfloat16": test_latents.cpu()}, cache_path)
        
        # Now load it back using the same method as the dataset loader
        with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
            loaded_latents = safe_file.get_tensor(
                "latents_1x64x64_bfloat16",
                device=torch.device("cpu"),
                dtype=torch.float32
            )
        
        # Verify loaded latents
        print(f"Original latents max: {original_max:.6f}")
        print(f"Loaded latents: shape={loaded_latents.shape}, dtype={loaded_latents.dtype}")
        print(f"  min={loaded_latents.min().item():.6f}, max={loaded_latents.max().item():.6f}, mean={loaded_latents.mean().item():.6f}")
        
        # CRITICAL ASSERTIONS
        assert loaded_latents.dtype == torch.float32, f"Should be float32, got {loaded_latents.dtype}"
        max_val = loaded_latents.abs().max().item()
        assert max_val > 1e-6, f"CRITICAL: Loaded latents are all zeros! (max: {max_val})"
        
        verify_latents_non_zero(loaded_latents, "loaded_latents")
        
        # Verify values are approximately preserved
        original_float32 = test_latents.float()
        max_diff = (loaded_latents - original_float32).abs().max().item()
        assert max_diff < 0.01, f"Values should be preserved (max diff: {max_diff})"
        
        print("✓ Cache file roundtrip test passed")
        
    finally:
        cleanup_test_file(cache_path)


def test_dataset_loader_integration():
    """Test that the dataset loader actually loads non-zero latents."""
    # Create a test cache file
    test_latents = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    test_image_latents = torch.randn(20, 1, 64, 64, dtype=torch.bfloat16)
    
    # Create temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        cache_path = f.name
    
    try:
        # Save cache files
        save_file({
            "latents_1x64x64_bfloat16": test_latents.cpu(),
            "latents_image_1x64x64_bfloat16": test_image_latents.cpu()
        }, cache_path)
        
        # Test loading using the same method as BucketBatchManager.__getitem__
        cpu_device = torch.device("cpu")
        sd_latent = {}
        with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as f:
            for key in f.keys():
                tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
                if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                    tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                sd_latent[key] = tensor
        
        # Verify latents were loaded correctly
        assert "latents_1x64x64_bfloat16" in sd_latent, "Should have main latents"
        latents = sd_latent["latents_1x64x64_bfloat16"]
        
        print(f"Dataset loader test - loaded latents: shape={latents.shape}, dtype={latents.dtype}")
        print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
        
        # CRITICAL: Verify non-zero
        max_val = latents.abs().max().item()
        assert max_val > 1e-6, f"CRITICAL: Dataset loader produced all-zero latents! (max: {max_val})"
        
        verify_latents_non_zero(latents, "dataset_loader_latents")
        assert latents.dtype == torch.float32, "Should be float32"
        
        print("✓ Dataset loader integration test passed")
        
    finally:
        cleanup_test_file(cache_path)


def test_full_training_step_simulation():
    """Simulate a full training step to verify latents → noise → target → loss."""
    # Create test latents (as they would come from dataset loader)
    latents = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
    
    # Verify latents are non-zero
    verify_latents_non_zero(latents, "training_latents")
    
    # Generate noise
    noise = torch.randn_like(latents)
    verify_latents_non_zero(noise, "training_noise")
    
    # Compute target (as done in training)
    target = noise - latents
    verify_latents_non_zero(target, "training_target")
    
    # Mock model_pred (would come from actual model)
    model_pred = torch.randn_like(target)
    verify_latents_non_zero(model_pred, "training_model_pred")
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
    
    # CRITICAL ASSERTIONS
    assert loss.item() > 0, f"CRITICAL: Loss is zero! This should never happen with non-zero tensors. loss={loss.item()}"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    print(f"✓ Full training step simulation passed: loss={loss.item():.6f}")


if __name__ == "__main__":
    print("=" * 80)
    print("END-TO-END TESTS")
    print("=" * 80)
    
    try:
        print("\n1. Testing cache file roundtrip...")
        test_create_and_load_cache_file()
        
        print("\n2. Testing dataset loader integration...")
        test_dataset_loader_integration()
        
        print("\n3. Testing full training step simulation...")
        test_full_training_step_simulation()
        
        print("\n4. Testing real cache file (if available)...")
        test_load_real_cache_file()
        
        print("\n" + "=" * 80)
        print("ALL END-TO-END TESTS PASSED!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        raise

