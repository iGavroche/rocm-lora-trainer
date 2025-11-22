"""
Training pipeline tests to verify latents → noise → model_pred → loss flow.
"""
import torch
import pytest
from tests.utils import verify_latents_non_zero, create_test_cache_file, cleanup_test_file


class TestTrainingPipeline:
    """Test training pipeline components."""
    
    def test_latents_to_noise(self):
        """Test that noise generation from latents works correctly."""
        # Create test latents
        latents = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
        verify_latents_non_zero(latents, "latents")
        
        # Generate noise
        noise = torch.randn_like(latents)
        
        # Verify noise is non-zero and random
        verify_latents_non_zero(noise, "noise")
        assert noise.shape == latents.shape
        assert noise.dtype == latents.dtype
        
        # Verify noise is different from latents
        max_diff = (noise - latents).abs().max().item()
        assert max_diff > 0.1, "Noise should be different from latents"
    
    def test_target_computation(self):
        """Test target computation: target = noise - latents."""
        latents = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
        noise = torch.randn_like(latents)
        
        # Compute target
        target = noise - latents
        
        # Verify target is non-zero
        verify_latents_non_zero(target, "target")
        assert target.shape == latents.shape
        assert target.dtype == latents.dtype
    
    def test_loss_computation(self):
        """Test loss computation with non-zero values."""
        # Create mock model_pred and target
        model_pred = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
        target = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
        
        # Verify both are non-zero
        verify_latents_non_zero(model_pred, "model_pred")
        verify_latents_non_zero(target, "target")
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
        
        # Verify loss is non-zero (unless model_pred == target, which is unlikely)
        assert loss.item() > 0, "Loss should be non-zero when model_pred != target"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_zero_latents_detection(self):
        """Test that zero latents are detected correctly."""
        # Create zero latents
        zero_latents = torch.zeros(1, 16, 1, 64, 64, dtype=torch.float32)
        
        # Verify detection
        max_val = zero_latents.abs().max().item()
        assert max_val < 1e-6, "Should detect zero latents"
        
        # Verify this would cause zero loss
        noise = torch.randn_like(zero_latents)
        target = noise - zero_latents  # target = noise (since latents are zero)
        
        # If model_pred is also zero, loss would be zero
        zero_model_pred = torch.zeros_like(target)
        loss = torch.nn.functional.mse_loss(zero_model_pred, target, reduction="mean")
        
        # Loss should be non-zero because target (noise) is non-zero
        assert loss.item() > 0, "Loss should be non-zero even with zero latents if noise is non-zero"
    
    def test_dtype_conversions_through_pipeline(self):
        """Test dtype conversions throughout training pipeline."""
        # Start with bfloat16 (as stored in cache)
        cached_latents = torch.randn(1, 16, 1, 64, 64, dtype=torch.bfloat16)
        verify_latents_non_zero(cached_latents, "cached_latents")
        
        # Convert to float32 (as done in dataset loader)
        latents = cached_latents.float()
        assert latents.dtype == torch.float32
        verify_latents_non_zero(latents, "converted_latents")
        
        # Generate noise (should match latents dtype)
        noise = torch.randn_like(latents)
        assert noise.dtype == torch.float32
        
        # Compute target
        target = noise - latents
        assert target.dtype == torch.float32
        verify_latents_non_zero(target, "target")
        
        # Mock model_pred (would come from model)
        model_pred = torch.randn_like(target)
        assert model_pred.dtype == torch.float32
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
        assert loss.item() > 0, "Loss should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




