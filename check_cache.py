#!/usr/bin/env python3
"""Quick script to check if cache files contain valid (non-zero) latents"""
from safetensors import safe_open
import sys
import os

cache_file = sys.argv[1] if len(sys.argv) > 1 else "myface/image0001_0512x0512_wan.safetensors"

if not os.path.exists(cache_file):
    print(f"File not found: {cache_file}")
    sys.exit(1)

with safe_open(cache_file, framework="pt") as f:
    keys = list(f.keys())
    print(f"Keys in cache file: {keys}")
    
    for key in keys:
        tensor = f.get_tensor(key)
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        print(f"\n{key}:")
        print(f"  shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  min: {min_val:.6f}")
        print(f"  max: {max_val:.6f}")
        print(f"  mean: {mean_val:.6f}")
        
        if abs(min_val) < 1e-6 and abs(max_val) < 1e-6 and abs(mean_val) < 1e-6:
            print(f"  WARNING: This tensor is all zeros!")
        else:
            print(f"  OK: Contains non-zero values")





