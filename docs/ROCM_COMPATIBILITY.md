# ROCm Compatibility Guide for Musubi Tuner

## Overview

This document describes ROCm (AMD GPU) compatibility issues and workarounds for Musubi Tuner, particularly for gfx1151 (Strix Halo) architecture.

## Known Issues

### 1. BFloat16 to Float32 Conversion Bug

**Problem**: Direct bfloat16→float32 conversion on ROCm can produce zeros instead of preserving values.

**Root Cause**: 
- ROCm has a known bug where converting bfloat16 tensors to float32 can result in zero values
- This affects gfx1151 (Strix Halo) and potentially other RDNA 3.5 architectures
- NumPy doesn't support bfloat16, so `.numpy()` conversion is not possible

**Solution**: 
- Use `.float()` method on CPU instead of `.to(torch.float32)`
- Ensure tensors are on CPU before conversion
- Load bfloat16 tensors from safetensors and convert to float32 during loading

**Implementation**: 
- Fixed in `src/musubi_tuner/utils/safetensors_utils.py`
- Uses `tensor.float()` on CPU for bfloat16→float32 conversion
- Applied in both large tensor (>10MB) and small tensor loading paths

### 2. VAE Encoding Issues

**Problem**: VAE encoding can produce all-zero latents or Inf values on ROCm.

**Root Cause**:
- Implicit dtype conversions during VAE encoding
- `.float()` on bfloat16 tensors can trigger ROCm bug

**Solution**:
- Keep latents in bfloat16 during VAE encoding
- Avoid implicit conversions to float32
- Clamp Inf/NaN values when saving to cache

**Implementation**:
- Fixed in `src/musubi_tuner/wan/modules/vae.py`
- Changed `.float().squeeze(0)` to `.to(self.dtype).squeeze(0)`

### 3. Cache Save/Load Issues

**Problem**: Bfloat16 tensors become zeros when saved to or loaded from cache files.

**Root Cause**:
- Moving bfloat16 tensors to CPU can trigger conversion bugs
- Loading bfloat16 from safetensors and converting can fail

**Solution**:
- Convert to float32 before moving to CPU when saving
- Load as float32 directly from safetensors
- Use `MemoryEfficientSafeOpen` with explicit dtype conversion

**Implementation**:
- Fixed in `src/musubi_tuner/dataset/image_video_dataset.py`
- Uses `MemoryEfficientSafeOpen` to load latents as float32
- Converts bfloat16→float32 during loading, not after

## Compatibility Matrix

### Dtype Conversions on ROCm

| From | To | Method | Works? | Notes |
|------|-----|--------|--------|-------|
| bfloat16 | float32 | `.to(torch.float32)` on GPU | ❌ | Produces zeros |
| bfloat16 | float32 | `.float()` on CPU | ✅ | Recommended |
| bfloat16 | float32 | `.to(torch.float32)` on CPU | ✅ | Works but `.float()` preferred |
| bfloat16 | float16 | `.half()` | ✅ | Works |
| float32 | bfloat16 | `.to(torch.bfloat16)` | ✅ | Works |
| float16 | bfloat16 | `.to(torch.bfloat16)` | ✅ | Works |

### Architecture Support

| Architecture | ROCm Support | BFloat16 Support | Notes |
|--------------|--------------|------------------|-------|
| gfx1151 (Strix Halo) | ✅ ROCm 7.1.0+ | ⚠️ Partial | Precision issues, use workarounds |
| gfx1150 (Strix Point) | ✅ ROCm 7.1.0+ | ⚠️ Partial | Similar to gfx1151 |
| RDNA 3 (GFX11) | ✅ | ❌ | BFloat16 disabled due to precision issues |
| RDNA 4 (GFX12) | ✅ | ✅ | Full BFloat16 support |

## Workarounds

### 1. Loading BFloat16 Tensors from Safetensors

```python
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

# Load directly as float32 to avoid ROCm bug
with MemoryEfficientSafeOpen(cache_path) as safe_file:
    tensor = safe_file.get_tensor(
        "latents_1x64x64_bfloat16",
        device=torch.device("cpu"),
        dtype=torch.float32  # Convert during loading
    )
```

### 2. Converting BFloat16 to Float32

```python
# ✅ Correct: Use .float() on CPU
tensor_cpu = tensor.cpu()
tensor_float32 = tensor_cpu.float()

# ❌ Incorrect: Direct conversion on GPU
tensor_float32 = tensor.to(torch.float32)  # May produce zeros on ROCm
```

### 3. Saving BFloat16 Tensors to Cache

```python
# Convert to float32 before moving to CPU
if tensor.dtype == torch.bfloat16:
    tensor_cpu = tensor.detach().to(torch.float32).cpu().to(torch.bfloat16)
else:
    tensor_cpu = tensor.detach().cpu()
```

## Testing

Comprehensive tests are available in the `tests/` directory:

- `tests/test_bfloat16_fix.py`: Minimal test to verify the fix works
- `tests/test_dtype_conversions.py`: Unit tests for all dtype conversions
- `tests/test_dataset_loading.py`: Integration tests for dataset loading
- `tests/test_training_pipeline.py`: Training pipeline tests
- `tests/test_rocm_compatibility.py`: ROCm-specific compatibility tests

Run tests with:
```bash
python -m pytest tests/ -v
```

Or run individual test files:
```bash
python tests/test_bfloat16_fix.py
```

## Recommendations

1. **Always use float32 for latents during training**: While bfloat16 is more memory-efficient, float32 avoids ROCm conversion bugs and is acceptable for training.

2. **Load latents as float32**: When loading from cache, always specify `dtype=torch.float32` to avoid conversion bugs.

3. **Test dtype conversions**: Use the provided tests to verify dtype conversions work correctly on your system.

4. **Monitor for updates**: ROCm is actively developed. Check for updates that might fix these issues:
   - ROCm release notes: https://rocmdocs.amd.com/
   - PyTorch ROCm support: https://pytorch.org/get-started/locally/

## References

- [ROCm Documentation](https://rocmdocs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [AMD Strix Halo (gfx1151) Support](https://blog.patti.tech/2024/01/23/amd-strix-point-halo-gfx1151-strix-point-gfx1150-apus-spotted-in-rocm-rdna-3-5-igpu-confirmed/)

## Changelog

- **2024-01-XX**: Initial documentation
- **2024-01-XX**: Fixed bfloat16→float32 conversion using `.float()` on CPU
- **2024-01-XX**: Fixed VAE encoding to preserve bfloat16 values
- **2024-01-XX**: Added comprehensive test suite




