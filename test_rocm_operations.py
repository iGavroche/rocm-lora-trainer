#!/usr/bin/env python3
"""
Minimal test to isolate ROCm SIGSEGV issues on Strix Halo
Tests various PyTorch operations that might cause crashes
"""
import torch
import sys
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Test 1: Basic tensor operations
print("=" * 60)
print("TEST 1: Basic tensor creation and operations")
print("=" * 60)
try:
    latents = torch.randn(1, 16, 1, 64, 64, device=device, dtype=torch.float16)
    print(f"✓ Created latents: {latents.shape}, device: {latents.device}")
    
    # Test noise generation (the one we fixed)
    noise_cpu = torch.randn_like(latents.cpu())
    noise = noise_cpu.to(device=device, dtype=latents.dtype)
    print(f"✓ Generated noise: {noise.shape}, device: {noise.device}")
    
    # Test tensor arithmetic
    noisy = latents + noise
    print(f"✓ Tensor addition: {noisy.shape}")
    
    # Test scaling
    scaled = latents * 0.5
    print(f"✓ Tensor scaling: {scaled.shape}")
    
except Exception as e:
    print(f"✗ ERROR in TEST 1: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Noise scheduler operations (likely where get_noisy_model_input_and_timesteps crashes)
print("\n" + "=" * 60)
print("TEST 2: Noise scheduler operations")
print("=" * 60)
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from musubi_tuner.utils.device_utils import synchronize_device
    
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(50)
    
    latents = torch.randn(1, 16, 1, 64, 64, device=device, dtype=torch.float16)
    noise_cpu = torch.randn_like(latents.cpu())
    noise = noise_cpu.to(device=device, dtype=latents.dtype)
    
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
    print(f"✓ Created timesteps: {timesteps.shape}, device: {timesteps.device}")
    
    synchronize_device(device)
    
    # This is likely where it crashes - scaling noise by sigma
    sigmas = scheduler.sigmas[timesteps]
    print(f"✓ Retrieved sigmas: {sigmas.shape}, device: {sigmas.device}")
    
    synchronize_device(device)
    
    # Scale noise
    scaled_noise = noise * sigmas.view(-1, 1, 1, 1, 1)
    print(f"✓ Scaled noise: {scaled_noise.shape}")
    
    synchronize_device(device)
    
    # Add to latents
    noisy_input = latents + scaled_noise
    print(f"✓ Created noisy input: {noisy_input.shape}")
    
    synchronize_device(device)
    
except Exception as e:
    print(f"✗ ERROR in TEST 2: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Indexing operations (another common crash point)
print("\n" + "=" * 60)
print("TEST 3: Tensor indexing and view operations")
print("=" * 60)
try:
    from musubi_tuner.utils.device_utils import synchronize_device
    
    latents = torch.randn(1, 16, 1, 64, 64, device=device, dtype=torch.float16)
    timesteps = torch.tensor([100], device=device)
    
    synchronize_device(device)
    
    # Test indexing
    indexed = latents[0]
    print(f"✓ Indexed tensor: {indexed.shape}")
    
    synchronize_device(device)
    
    # Test view operations
    viewed = latents.view(1, -1)
    print(f"✓ Viewed tensor: {viewed.shape}")
    
    synchronize_device(device)
    
    # Test unsqueeze
    unsqueezed = latents.unsqueeze(0)
    print(f"✓ Unsqueezed tensor: {unsqueezed.shape}")
    
    synchronize_device(device)
    
except Exception as e:
    print(f"✗ ERROR in TEST 3: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Mixed precision operations
print("\n" + "=" * 60)
print("TEST 4: Mixed precision (autocast) operations")
print("=" * 60)
try:
    from musubi_tuner.utils.device_utils import synchronize_device
    
    latents = torch.randn(1, 16, 1, 64, 64, device=device, dtype=torch.float16)
    noise_cpu = torch.randn_like(latents.cpu())
    noise = noise_cpu.to(device=device, dtype=latents.dtype)
    
    synchronize_device(device)
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        result = latents + noise
        synchronize_device(device)
        result = result * 0.5
        synchronize_device(device)
    
    print(f"✓ Autocast operations completed: {result.shape}")
    
except Exception as e:
    print(f"✗ ERROR in TEST 4: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)







