#!/usr/bin/env python3
"""Test if VAE encoding works correctly"""
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from musubi_tuner.wan.modules.vae import WanVAE

# Load VAE
vae_path = "models/wan/wan_2.1_vae.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print(f"Loading VAE from {vae_path}")
vae = WanVAE(z_dim=16, vae_path=vae_path, dtype=dtype, device=device)
vae.eval()

# Check if VAE model loaded correctly
print(f"VAE model device: {vae.device}, dtype: {vae.dtype}")
print(f"VAE model mean shape: {vae.mean.shape}, std shape: {vae.std.shape}")
print(f"VAE scale: mean min={vae.scale[0].min().item():.4f}, max={vae.scale[0].max().item():.4f}, std min={vae.scale[1].min().item():.4f}, max={vae.scale[1].max().item():.4f}")

# Load a test image
test_image_path = "myface/image0001.jpg"
if not os.path.exists(test_image_path):
    print(f"Test image not found: {test_image_path}")
    sys.exit(1)

print(f"Loading test image: {test_image_path}")
img = Image.open(test_image_path).convert("RGB")
img_array = np.array(img)  # H, W, C

# Convert to tensor format expected by VAE: [C, F, H, W]
contents = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # 1, 1, H, W, C
contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # 1, C, F, H, W
contents = contents.to(vae.device, dtype=vae.dtype)
contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

print(f"Input shape: {contents.shape}, dtype: {contents.dtype}")
print(f"Input stats: min={contents.min().item():.4f}, max={contents.max().item():.4f}, mean={contents.mean().item():.4f}")

# Encode - test direct model.encode call
print("Encoding with VAE...")
print(f"Contents[0] shape: {contents[0].shape}, device: {contents[0].device}, dtype: {contents[0].dtype}")

# Test direct model.encode to see intermediate values
print("Testing direct model.encode...")
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=vae.dtype):
    # vae.encode calls model.encode with scale
    mu = vae.model.encode(contents[0].unsqueeze(0), vae.scale)  # model.encode expects [1, C, F, H, W]
    print(f"Model encode output (before float) shape: {mu.shape}, dtype: {mu.dtype}")
    print(f"Model encode output stats: min={mu.min().item():.6f}, max={mu.max().item():.6f}, mean={mu.mean().item():.6f}")
    print(f"Model encode output has NaN: {torch.isnan(mu).any().item()}, has Inf: {torch.isinf(mu).any().item()}")
    
    # Test the conversion
    mu_float = mu.float()
    print(f"After .float() shape: {mu_float.shape}, dtype: {mu_float.dtype}")
    print(f"After .float() stats: min={mu_float.min().item():.6f}, max={mu_float.max().item():.6f}, mean={mu_float.mean().item():.6f}")
    print(f"After .float() has NaN: {torch.isnan(mu_float).any().item()}, has Inf: {torch.isinf(mu_float).any().item()}")
    
    mu_squeezed = mu_float.squeeze(0)
    print(f"After .squeeze(0) shape: {mu_squeezed.shape}, dtype: {mu_squeezed.dtype}")
    print(f"After .squeeze(0) stats: min={mu_squeezed.min().item():.6f}, max={mu_squeezed.max().item():.6f}, mean={mu_squeezed.mean().item():.6f}")
    
    # Now test vae.encode (should use the fixed version)
    print("\nTesting vae.encode (with fix)...")
    latent = vae.encode([contents[0]])  # vae.encode expects list of [C, F, H, W]
    print(f"vae.encode output shape: {latent[0].shape}, dtype: {latent[0].dtype}")
    print(f"vae.encode output stats: min={latent[0].min().item():.6f}, max={latent[0].max().item():.6f}, mean={latent[0].mean().item():.6f}")

print(f"Encoded latent shape: {latent[0].shape}, dtype: {latent[0].dtype}")
print(f"Latent stats: min={latent[0].min().item():.6f}, max={latent[0].max().item():.6f}, mean={latent[0].mean().item():.6f}")

if latent[0].abs().max().item() < 1e-6:
    print("ERROR: VAE encoding produced all zeros!")
    sys.exit(1)
elif torch.isinf(latent[0]).any() or torch.isnan(latent[0]).any():
    print("ERROR: VAE encoding produced Inf or NaN values!")
    sys.exit(1)
else:
    print("SUCCESS: VAE encoding produced valid latents!")

