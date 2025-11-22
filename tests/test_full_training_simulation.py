"""
Full training simulation test - simulates the exact process used in training.
This test MUST pass for training to work.
"""
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def simulate_full_training_step(cache_file):
    """Simulate a full training step using the real cache file."""
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    print(f"Simulating full training step with: {cache_file}")
    
    # Step 1: Load latents (as dataset loader does)
    cpu_device = torch.device("cpu")
    sd_latent = {}
    
    with MemoryEfficientSafeOpen(cache_file, disable_numpy_memmap=False) as f:
        for key in f.keys():
            if "latents" in key:  # Only load latent keys
                tensor = f.get_tensor(key, device=cpu_device, dtype=torch.float32)
                
                # Apply clamping logic from dataset loader
                if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                    tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Clamp extreme values
                max_val = tensor.abs().max().item()
                if max_val > 1e10:
                    tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                
                sd_latent[key] = tensor
    
    # Step 2: Process keys and create batch (as BucketBatchManager does)
    batch_tensor_data = {}
    for key in sd_latent.keys():
        content_key = key.rsplit("_", 1)[0]  # remove dtype
        if content_key.startswith("latents_"):
            content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW
        
        if content_key not in batch_tensor_data:
            batch_tensor_data[content_key] = []
        batch_tensor_data[content_key].append(sd_latent[key])
    
    # Stack
    for key in batch_tensor_data.keys():
        batch_tensor_data[key] = torch.stack(batch_tensor_data[key])
    
    # Step 3: Get latents from batch (as training loop does)
    if "latents" not in batch_tensor_data:
        raise AssertionError("'latents' key not found in batch")
    
    latents = batch_tensor_data["latents"]
    
    print(f"\nStep 1 - Loaded latents: shape={latents.shape}, dtype={latents.dtype}")
    print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
    
    # CRITICAL CHECK
    max_val = latents.abs().max().item()
    if max_val < 1e-6:
        raise AssertionError(f"CRITICAL: Latents are all zeros after loading! max={max_val}")
    
    # Step 4: Scale/shift (if any - WAN doesn't do this, but check)
    # latents = scale_shift_latents(latents)  # WAN returns latents as-is
    
    print(f"Step 2 - After scale_shift: shape={latents.shape}, dtype={latents.dtype}")
    print(f"  min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}")
    
    max_val = latents.abs().max().item()
    if max_val < 1e-6:
        raise AssertionError(f"CRITICAL: Latents are all zeros after scale_shift! max={max_val}")
    
    # Step 5: Generate noise
    noise = torch.randn_like(latents)
    
    print(f"Step 3 - Generated noise: shape={noise.shape}, dtype={noise.dtype}")
    print(f"  min={noise.min().item():.6f}, max={noise.max().item():.6f}, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
    
    max_noise = noise.abs().max().item()
    if max_noise < 1e-6:
        raise AssertionError(f"CRITICAL: Noise is all zeros! max={max_noise}")
    
    # Step 6: Compute target
    target = noise - latents
    
    print(f"Step 4 - Target: shape={target.shape}, dtype={target.dtype}")
    print(f"  min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}")
    
    max_target = target.abs().max().item()
    if max_target < 1e-6:
        raise AssertionError(f"CRITICAL: Target is all zeros! max={max_target}")
    
    # Step 7: Mock model_pred (would come from actual model)
    model_pred = torch.randn_like(target)
    
    # Step 8: Compute loss
    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
    
    print(f"Step 5 - Loss: {loss.item():.6f}")
    
    if loss.item() < 1e-6:
        raise AssertionError(f"CRITICAL: Loss is zero! loss={loss.item()}")
    
    print("\n✓ Full training step simulation PASSED")
    print(f"  Final loss: {loss.item():.6f}")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("FULL TRAINING SIMULATION TEST")
    print("=" * 80)
    
    cache_file = "myface/image0001_0512x0512_wan.safetensors"
    
    try:
        success = simulate_full_training_step(cache_file)
        if success:
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED - Training simulation works correctly!")
            print("=" * 80)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




