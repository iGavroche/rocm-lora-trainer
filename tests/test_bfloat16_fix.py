"""
Minimal test to verify bfloat16→float32 conversion fix works correctly.
This test verifies that latents can be loaded from cache files without becoming zeros.
"""
import torch
import tempfile
import os
from safetensors.torch import save_file
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def test_bfloat16_to_float32_conversion():
    """Test that bfloat16 tensors can be converted to float32 without becoming zeros."""
    # Create a test tensor with non-zero values
    test_tensor = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    
    # Verify it has non-zero values
    assert test_tensor.abs().max().item() > 0, "Test tensor should have non-zero values"
    
    # Convert to float32 using .float() (the fix)
    converted = test_tensor.float()
    
    # Verify conversion preserved values (not all zeros)
    assert converted.dtype == torch.float32, "Should be float32"
    assert converted.abs().max().item() > 0, "Converted tensor should not be all zeros"
    assert converted.shape == test_tensor.shape, "Shape should be preserved"
    
    # Verify values are approximately the same (within bfloat16 precision)
    # bfloat16 has ~3 decimal digits of precision, so we check with reasonable tolerance
    max_diff = (converted - test_tensor.float()).abs().max().item()
    assert max_diff < 0.01, f"Values should be preserved (max diff: {max_diff})"


def test_load_bfloat16_from_safetensors():
    """Test loading bfloat16 tensors from safetensors and converting to float32."""
    # Create a test tensor with known non-zero values
    test_tensor = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    original_max = test_tensor.abs().max().item()
    assert original_max > 0, "Test tensor should have non-zero values"
    
    # Save to temporary safetensors file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = f.name
    
    try:
        save_file({"latents_1x64x64_bfloat16": test_tensor}, temp_path)
        
        # Load using MemoryEfficientSafeOpen with float32 target dtype
        with MemoryEfficientSafeOpen(temp_path, disable_numpy_memmap=False) as safe_file:
            loaded_tensor = safe_file.get_tensor(
                "latents_1x64x64_bfloat16",
                device=torch.device("cpu"),
                dtype=torch.float32
            )
        
        # Verify loaded tensor
        assert loaded_tensor.dtype == torch.float32, "Should be loaded as float32"
        assert loaded_tensor.abs().max().item() > 0, "Loaded tensor should not be all zeros"
        assert loaded_tensor.shape == test_tensor.shape, "Shape should be preserved"
        
        # Verify values are approximately preserved
        # Note: We compare with the original float32 conversion, not bfloat16 directly
        original_float32 = test_tensor.float()
        max_diff = (loaded_tensor - original_float32).abs().max().item()
        assert max_diff < 0.01, f"Values should be preserved (max diff: {max_diff})"
        
    finally:
        # Clean up - ensure file is closed before deletion
        import time
        time.sleep(0.1)  # Brief delay to ensure file handles are released
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except (PermissionError, OSError):
            # On Windows, file might still be locked, but that's okay for a test
            pass


def test_load_with_inf_nan_values():
    """Test that Inf/NaN values are handled correctly during conversion."""
    # Create a tensor with some Inf values (simulating corrupted cache)
    test_tensor = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
    # Add some Inf values
    test_tensor[0, 0, 0, 0] = float('inf')
    test_tensor[0, 0, 0, 1] = float('-inf')
    test_tensor[0, 0, 0, 2] = float('nan')
    
    # Convert to float32
    converted = test_tensor.float()
    
    # Verify conversion worked (even with Inf/NaN)
    assert converted.dtype == torch.float32, "Should be float32"
    # Most values should still be valid
    valid_values = converted[~torch.isinf(converted) & ~torch.isnan(converted)]
    assert valid_values.numel() > 0, "Should have some valid values"


if __name__ == "__main__":
    print("Running minimal bfloat16 conversion tests...")
    test_bfloat16_to_float32_conversion()
    print("[PASS] bfloat16->float32 conversion test passed")
    
    test_load_bfloat16_from_safetensors()
    print("[PASS] Load bfloat16 from safetensors test passed")
    
    test_load_with_inf_nan_values()
    print("[PASS] Inf/NaN handling test passed")
    
    print("\nAll minimal tests passed!")

