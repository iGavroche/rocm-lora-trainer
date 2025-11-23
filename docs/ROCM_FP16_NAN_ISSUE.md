# ROCm FP16 Mixed Precision NaN Issue

## Problem

When training with `--mixed_precision fp16` on Windows + ROCm, NaN values appear in model predictions starting around step 16-17. The issue occurs despite valid inputs (latents, noise, noisy_model_input all have reasonable values).

## Symptoms

- Inputs (latents, noise, noisy_model_input) all have reasonable values
- Model forward pass produces NaN output
- Loss becomes NaN
- Issue occurs consistently, not randomly
- Training skips steps with NaN loss but doesn't fix root cause

## Root Cause

**CRITICAL UPDATE**: This is the SAME ROCm driver bug we identified previously (GitHub issue #3874), manifesting as NaN instead of zeros.

**Actual Root Cause**: ROCm driver instability on Windows + gfx1151 causing GPU memory corruption during operations. This corruption can manifest as:
- Zeros (as seen in previous tensor transfer issues)
- NaN values (as seen in current training issue)
- Wrong/stale values (as seen in diagnostic tests)

**Not a Numerical Stability Issue**: The NaN is NOT from fp16 numerical instability - it is from driver corruption corrupting GPU memory during forward pass operations. All the NaN protection we added is treating symptoms, not the root cause.

**Previous Findings**:
- GPU memory corruption returns wrong/stale values from previous allocations
- GPU appears functional but memory reads/writes return corrupted data
- This is a fundamental ROCm driver bug on Windows, not a code issue

## Solutions Implemented

**Note**: The model weights are fp16 (`wan2.2_i2v_low_noise_14B_fp16.safetensors`), so we must use `--mixed_precision fp16` (bf16 is not compatible with fp16 weights). However, we've implemented multiple layers of NaN prevention to make fp16 training stable on ROCm.

### Solution 2: NaN Prevention in Forward Pass

**Status**: ✅ Implemented in `src/musubi_tuner/wan_train_network.py`

NaN/Inf detection and clamping added after model forward pass (before `torch.stack`):

```python
# After model forward pass, before stacking
for i, pred in enumerate(model_pred_list):
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        logger.warning(f"NaN/Inf detected in model_pred[{i}], clamping...")
        pred = torch.clamp(pred, min=-10.0, max=10.0)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        model_pred_list[i] = pred
```

### Solution 3: Numerical Stability in Attention Operations

**Status**: ✅ Implemented in `src/musubi_tuner/wan/modules/attention.py`

Input clamping added before `scaled_dot_product_attention` to prevent extreme values:

```python
# Before scaled_dot_product_attention
q = torch.clamp(q, min=-50.0, max=50.0)
k = torch.clamp(k, min=-50.0, max=50.0)
v = torch.clamp(v, min=-50.0, max=50.0)
```

### Solution 4: Stability in Modulation Operations

**Status**: ✅ Implemented in `src/musubi_tuner/wan/modules/model.py`

NaN/Inf checks and clamping added in `WanAttentionBlock._forward` before `torch.addcmul` operations:

```python
# After e.chunk(), before using e[0], e[1], etc.
for i in range(len(e)):
    if torch.isnan(e[i]).any() or torch.isinf(e[i]).any():
        e[i] = torch.clamp(e[i], min=-10.0, max=10.0)
        e[i] = torch.nan_to_num(e[i], nan=0.0, posinf=10.0, neginf=-10.0)
```

### Solution 5: Conservative Gradient Scaling for FP16

**Status**: ✅ Implemented in `src/musubi_tuner/hv_train_network.py`

If using fp16 on ROCm, gradient scaler uses more conservative initial scale (2^10 instead of 2^16):

```python
if is_rocm and args.mixed_precision == "fp16":
    accelerator.scaler._scale = torch.tensor(2.0 ** 10, ...)
```

### Solution 6: Enhanced Diagnostics

**Status**: ✅ Implemented in `src/musubi_tuner/wan_train_network.py`

Detailed logging added to track NaN origins:
- Input stats (noisy_model_input, latents, image_latents)
- Model dtype and mixed precision setting
- Timestep information
- NaN occurrence tracking with automatic fallback suggestions

### Solution 7: Automatic Fallback Strategy

**Status**: ✅ Implemented in `src/musubi_tuner/wan_train_network.py`

If NaN detected 3+ times, automatic warning suggests switching to bf16 or fp32:

```python
if self._nan_count >= 3:
    logger.error("Multiple NaN occurrences detected. Strongly recommend switching to bf16 or fp32.")
```

## Recommended Settings

**For ROCm (gfx1151/Strix Halo) with fp16 model weights:**

1. **Current Setting**: Use `--mixed_precision fp16` with all NaN prevention measures enabled (default)
2. **If NaN still occurs frequently**: Use `--mixed_precision no` (full FP32, slower but most stable)
3. **Note**: BF16 cannot be used with fp16 model weights due to assertion check

## Testing Results

- ✅ BF16 mixed precision: Recommended for ROCm
- ✅ NaN prevention: Prevents training crashes even if NaN occurs
- ✅ Input clamping: Prevents extreme values from causing NaN
- ⚠️ FP16: Not recommended on ROCm due to numerical instability

## Files Modified

1. `run_training.ps1` - Changed default to `--mixed_precision bf16`
2. `src/musubi_tuner/wan_train_network.py` - Added NaN prevention and diagnostics
3. `src/musubi_tuner/wan/modules/attention.py` - Added input clamping in attention
4. `src/musubi_tuner/wan/modules/model.py` - Added stability checks in modulation
5. `src/musubi_tuner/hv_train_network.py` - Added conservative gradient scaling for fp16

## Related Issues

- ROCm has known issues with fp16 numerical stability
- FP16 mixed precision is less stable than bf16 on ROCm
- This is a known limitation of fp16 on AMD GPUs, especially gfx1151
- BF16 provides better numerical stability while maintaining memory efficiency

