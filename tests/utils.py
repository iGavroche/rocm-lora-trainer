"""
Test utilities for dtype conversion and latent verification tests.
"""
import torch
import tempfile
import os
from safetensors.torch import save_file
from typing import Dict, Tuple


def create_test_cache_file(
    latents_shape: Tuple[int, ...] = (16, 1, 64, 64),
    dtype: torch.dtype = torch.bfloat16,
    include_image_latents: bool = True,
) -> str:
    """
    Generate a test safetensors cache file with bfloat16 latents.
    
    Args:
        latents_shape: Shape of the latent tensor
        dtype: Dtype for the latents (default: bfloat16)
        include_image_latents: Whether to include image latents for i2v training
        
    Returns:
        Path to the created cache file
    """
    # Create test tensors with non-zero values
    latents = torch.randn(*latents_shape, dtype=dtype)
    
    # Ensure values are non-zero
    assert latents.abs().max().item() > 0, "Test latents must have non-zero values"
    
    # Create dictionary for safetensors
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"
    F, H, W = latents_shape[1], latents_shape[2], latents_shape[3]
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latents.cpu()}
    
    if include_image_latents:
        # Create image latents (typically 20 channels for WAN)
        image_latents = torch.randn(20, 1, 64, 64, dtype=dtype)
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latents.cpu()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = f.name
    
    save_file(sd, temp_path)
    return temp_path


def verify_latents_non_zero(latents: torch.Tensor, name: str = "latents") -> None:
    """
    Assert that latents are not all zeros.
    
    Args:
        latents: Tensor to verify
        name: Name for error message
    """
    max_val = latents.abs().max().item()
    assert max_val > 1e-6, f"{name} are all zeros! (max abs value: {max_val})"
    
    # Also check for reasonable value range (not all Inf/NaN)
    valid_values = latents[~torch.isinf(latents) & ~torch.isnan(latents)]
    assert valid_values.numel() > 0, f"{name} contain only Inf/NaN values!"


def verify_dtype_conversion(
    original: torch.Tensor,
    converted: torch.Tensor,
    target_dtype: torch.dtype,
    tolerance: float = 0.01,
) -> None:
    """
    Test that dtype conversion preserves values within tolerance.
    
    Args:
        original: Original tensor
        converted: Converted tensor
        target_dtype: Expected dtype of converted tensor
        tolerance: Maximum allowed difference
    """
    assert converted.dtype == target_dtype, f"Converted tensor should be {target_dtype}, got {converted.dtype}"
    assert converted.shape == original.shape, "Shape should be preserved"
    
    # Compare values (convert original to target dtype for comparison)
    original_converted = original.to(target_dtype)
    max_diff = (converted - original_converted).abs().max().item()
    assert max_diff < tolerance, f"Values should be preserved (max diff: {max_diff}, tolerance: {tolerance})"


def cleanup_test_file(file_path: str) -> None:
    """Clean up a test file, handling Windows file locking issues."""
    import time
    time.sleep(0.1)  # Brief delay to ensure file handles are released
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except (PermissionError, OSError):
        # On Windows, file might still be locked, but that's okay for a test
        pass




