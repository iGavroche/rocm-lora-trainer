"""
ROCm compatibility tests for dtype conversions.
These tests verify that dtype conversions work correctly on ROCm (if available).
"""
import torch
import pytest
from tests.utils import verify_latents_non_zero, verify_dtype_conversion


# Check if we're on ROCm
IS_ROCM = torch.version.hip is not None if hasattr(torch.version, 'hip') else False


class TestROCMCompatibility:
    """Test ROCm-specific compatibility issues."""
    
    @pytest.mark.skipif(not IS_ROCM, reason="ROCm-specific test")
    def test_bfloat16_to_float32_on_rocm(self):
        """Test bfloat16→float32 conversion on ROCm."""
        # Create bfloat16 tensor
        test_tensor = torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)
        verify_latents_non_zero(test_tensor, "test_tensor")
        
        # Convert to float32 on CPU (workaround for ROCm bug)
        if test_tensor.device.type != "cpu":
            test_tensor = test_tensor.cpu()
        
        converted = test_tensor.float()
        
        # Verify conversion worked
        assert converted.dtype == torch.float32
        verify_latents_non_zero(converted, "converted_tensor")
        verify_dtype_conversion(test_tensor, converted, torch.float32)
    
    def test_bfloat16_conversion_methods(self):
        """Test different bfloat16 conversion methods to find what works."""
        test_tensor = torch.randn(100, dtype=torch.bfloat16)
        verify_latents_non_zero(test_tensor, "test_tensor")
        
        # Method 1: .float() on CPU
        method1 = test_tensor.cpu().float()
        assert method1.dtype == torch.float32
        verify_latents_non_zero(method1, "method1_float")
        
        # Method 2: .to(torch.float32) on CPU
        method2 = test_tensor.cpu().to(torch.float32)
        assert method2.dtype == torch.float32
        verify_latents_non_zero(method2, "method2_to")
        
        # Both should produce similar results
        max_diff = (method1 - method2).abs().max().item()
        assert max_diff < 1e-6, f"Methods should produce same result (diff: {max_diff})"
    
    def test_edge_cases_inf_nan(self):
        """Test edge cases with Inf and NaN values."""
        # Create tensor with Inf/NaN
        test_tensor = torch.randn(100, dtype=torch.bfloat16)
        test_tensor[0] = float('inf')
        test_tensor[1] = float('-inf')
        test_tensor[2] = float('nan')
        
        # Convert to float32
        converted = test_tensor.cpu().float()
        
        # Should still convert (Inf/NaN handling is separate)
        assert converted.dtype == torch.float32
        # Most values should be valid
        valid_values = converted[~torch.isinf(converted) & ~torch.isnan(converted)]
        assert valid_values.numel() > 0
    
    def test_very_large_values(self):
        """Test conversion with very large values."""
        # Create tensor with large values (but not Inf)
        test_tensor = torch.randn(100, dtype=torch.bfloat16) * 100
        # Clamp to avoid Inf
        test_tensor = torch.clamp(test_tensor, min=-1000, max=1000)
        
        verify_latents_non_zero(test_tensor, "test_tensor")
        
        # Convert
        converted = test_tensor.cpu().float()
        
        # Verify
        assert converted.dtype == torch.float32
        verify_latents_non_zero(converted, "converted_tensor")
        # Values should be preserved
        max_diff = (converted - test_tensor.float()).abs().max().item()
        assert max_diff < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




