# ROCm FP16 Mixed Precision NaN Issue

## Problem

When training with `--mixed_precision fp16` on Windows + ROCm, NaN values appear in model predictions starting around step 100.

## Symptoms

- Inputs (latents, noise, noisy_model_input) all have reasonable values
- Model forward pass produces NaN output
- Loss becomes NaN
- Issue occurs consistently, not randomly
- Training continues but produces invalid results

## Root Cause

Likely numerical instability in fp16 mixed precision on ROCm. FP16 has a smaller dynamic range than FP32/BF16, and certain operations in the model forward pass may overflow/underflow, producing NaN.

## Potential Solutions

### Option 1: Use BF16 Instead of FP16 (Recommended)

BF16 has the same dynamic range as FP32 but uses less memory. It's more stable than FP16:

```powershell
# In run_training.ps1, change:
--mixed_precision fp16
# To:
--mixed_precision bf16
```

**Note**: You'll need to ensure your model weights support bf16. The WAN model you're using (`wan2.2_i2v_low_noise_14B_fp16.safetensors`) is in fp16, so you may need to convert it or use a bf16 version.

### Option 2: Disable Mixed Precision

Train in full FP32 (slower but most stable):

```powershell
--mixed_precision no
```

### Option 3: Add Numerical Stability Checks

Add NaN detection and replacement in the model forward pass (may mask underlying issues):

```python
# After model forward pass
if torch.isnan(model_pred).any():
    logger.warning("NaN detected, replacing with zeros")
    model_pred = torch.nan_to_num(model_pred, nan=0.0)
```

**Warning**: This may mask the real issue and produce incorrect training results.

## Current Status

- Issue logged to memory MCP
- Training continues but produces NaN loss
- Need to test with bf16 or no mixed precision

## Related Issues

- ROCm has known issues with fp16 numerical stability
- FP16 mixed precision is less stable than bf16 on ROCm
- This is a known limitation of fp16 on AMD GPUs

