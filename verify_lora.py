#!/usr/bin/env python3
"""
Verify LoRA weights are valid and not corrupted.
Checks for:
- All zeros
- NaN/Inf values
- Empty weights
- Key structure
"""

import sys
import torch
from safetensors.torch import load_file

def verify_lora(lora_file):
    print(f"Checking LoRA file: {lora_file}")
    print("=" * 60)
    
    try:
        sd = load_file(lora_file)
    except Exception as e:
        print(f"ERROR: Could not load file: {e}")
        return False
    
    keys = list(sd.keys())
    if len(keys) == 0:
        print("ERROR: LoRA file has no keys!")
        return False
    
    print(f"Total keys: {len(keys)}")
    print(f"\nFirst 10 keys:")
    for k in keys[:10]:
        print(f"  {k}")
    
    # Check key structure
    has_lora_unet = any("lora_unet" in k for k in keys)
    has_lora_down = any("lora_down" in k for k in keys)
    has_lora_up = any("lora_up" in k for k in keys)
    has_alpha = any("alpha" in k for k in keys)
    
    print(f"\nKey structure:")
    print(f"  Has 'lora_unet' prefix: {has_lora_unet}")
    print(f"  Has 'lora_down' keys: {has_lora_down}")
    print(f"  Has 'lora_up' keys: {has_lora_up}")
    print(f"  Has 'alpha' keys: {has_alpha}")
    
    # Check for all zeros, NaN, Inf
    all_zeros = True
    has_nan = False
    has_inf = False
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    
    print(f"\nChecking weights...")
    for key, tensor in sd.items():
        if "alpha" in key:
            continue  # Skip alpha, it's a scalar
        
        total_params += tensor.numel()
        
        # Check for zeros
        zero_count = (tensor == 0).sum().item()
        zero_params += zero_count
        if zero_count < tensor.numel():
            all_zeros = False
        
        # Check for NaN
        nan_count = torch.isnan(tensor).sum().item()
        nan_params += nan_count
        if nan_count > 0:
            has_nan = True
            print(f"  WARNING: {key} has {nan_count} NaN values!")
        
        # Check for Inf
        inf_count = torch.isinf(tensor).sum().item()
        inf_params += inf_count
        if inf_count > 0:
            has_inf = True
            print(f"  WARNING: {key} has {inf_count} Inf values!")
    
    print(f"\nWeight statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Zero parameters: {zero_params:,} ({100*zero_params/total_params:.2f}%)")
    print(f"  NaN parameters: {nan_params:,}")
    print(f"  Inf parameters: {inf_params:,}")
    
    # Check weight ranges
    all_tensors = [t for k, t in sd.items() if "alpha" not in k]
    if all_tensors:
        all_combined = torch.cat([t.flatten() for t in all_tensors])
        print(f"\nWeight value ranges:")
        print(f"  Min: {all_combined.min().item():.6f}")
        print(f"  Max: {all_combined.max().item():.6f}")
        print(f"  Mean: {all_combined.mean().item():.6f}")
        print(f"  Std: {all_combined.std().item():.6f}")
    
    # Final verdict
    print(f"\n" + "=" * 60)
    if all_zeros:
        print("❌ FAIL: All weights are zero! LoRA is not trained.")
        return False
    elif has_nan:
        print("❌ FAIL: LoRA contains NaN values! Training may have failed.")
        return False
    elif has_inf:
        print("FAIL: LoRA contains Inf values! Training may have failed.")
        return False
    elif zero_params / total_params > 0.99:
        print("⚠️  WARNING: More than 99% of weights are zero! LoRA may not be effective.")
        return True
    elif not has_lora_unet:
        print("⚠️  WARNING: LoRA keys don't have 'lora_unet' prefix. May not be compatible with WAN models.")
        return True
    else:
        print("✅ LoRA appears to be valid!")
        return True

if __name__ == "__main__":
    lora_file = sys.argv[1] if len(sys.argv) > 1 else "chani_i2v.lora.safetensors"
    success = verify_lora(lora_file)
    sys.exit(0 if success else 1)

