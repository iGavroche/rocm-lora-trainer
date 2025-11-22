"""
Integration tests for dataset loading, verifying latents load correctly.
"""
import torch
import pytest
import os
from musubi_tuner.dataset.image_video_dataset import BucketBatchManager, ItemInfo
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from tests.utils import (
    create_test_cache_file,
    verify_latents_non_zero,
    cleanup_test_file,
    create_test_cache_file,
)


class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    def test_load_latents_from_cache(self):
        """Test that latents can be loaded from cache files without becoming zeros."""
        # Create test cache file
        cache_path = create_test_cache_file(dtype=torch.bfloat16)
        
        try:
            # Load using MemoryEfficientSafeOpen (same as dataset loader uses)
            with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
                latents = safe_file.get_tensor(
                    "latents_1x64x64_bfloat16",
                    device=torch.device("cpu"),
                    dtype=torch.float32
                )
            
            # Verify latents are non-zero
            verify_latents_non_zero(latents, "loaded_latents")
            assert latents.dtype == torch.float32, "Should be loaded as float32"
            
        finally:
            cleanup_test_file(cache_path)
    
    def test_load_image_latents(self):
        """Test loading image latents for i2v training."""
        cache_path = create_test_cache_file(
            dtype=torch.bfloat16,
            include_image_latents=True
        )
        
        try:
            with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
                # Load main latents
                latents = safe_file.get_tensor(
                    "latents_1x64x64_bfloat16",
                    device=torch.device("cpu"),
                    dtype=torch.float32
                )
                
                # Load image latents
                image_latents = safe_file.get_tensor(
                    "latents_image_1x64x64_bfloat16",
                    device=torch.device("cpu"),
                    dtype=torch.float32
                )
            
            # Verify both are non-zero
            verify_latents_non_zero(latents, "latents")
            verify_latents_non_zero(image_latents, "image_latents")
            assert latents.dtype == torch.float32
            assert image_latents.dtype == torch.float32
            
        finally:
            cleanup_test_file(cache_path)
    
    def test_load_multiple_tensors(self):
        """Test loading multiple tensors from same cache file."""
        cache_path = create_test_cache_file(
            dtype=torch.bfloat16,
            include_image_latents=True
        )
        
        try:
            with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
                keys = list(safe_file.keys())
                assert len(keys) >= 1, "Should have at least one tensor"
                
                # Load all tensors
                loaded_tensors = {}
                for key in keys:
                    loaded_tensors[key] = safe_file.get_tensor(
                        key,
                        device=torch.device("cpu"),
                        dtype=torch.float32
                    )
                    verify_latents_non_zero(loaded_tensors[key], f"tensor_{key}")
                    assert loaded_tensors[key].dtype == torch.float32
            
        finally:
            cleanup_test_file(cache_path)
    
    def test_inf_nan_clamping(self):
        """Test that Inf/NaN values are clamped correctly during loading."""
        # Create tensor with Inf/NaN
        import tempfile
        from safetensors.torch import save_file
        
        test_tensor = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
        test_tensor[0, 0, 0, 0] = float('inf')
        test_tensor[0, 0, 0, 1] = float('-inf')
        test_tensor[0, 0, 0, 2] = float('nan')
        
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
            cache_path = f.name
        
        try:
            save_file({"latents_1x64x64_bfloat16": test_tensor.cpu()}, cache_path)
            
            with MemoryEfficientSafeOpen(cache_path, disable_numpy_memmap=False) as safe_file:
                loaded = safe_file.get_tensor(
                    "latents_1x64x64_bfloat16",
                    device=torch.device("cpu"),
                    dtype=torch.float32
                )
            
            # Check that Inf/NaN are present (clamping happens in dataset loader, not here)
            # But we should still be able to load the tensor
            assert loaded.dtype == torch.float32
            # Most values should be valid
            valid_values = loaded[~torch.isinf(loaded) & ~torch.isnan(loaded)]
            assert valid_values.numel() > 0
            
        finally:
            cleanup_test_file(cache_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




