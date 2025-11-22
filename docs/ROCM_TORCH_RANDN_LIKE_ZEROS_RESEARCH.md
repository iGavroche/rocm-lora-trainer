# ROCm torch.randn_like Producing Zeros - Research Findings

## Problem Summary
`torch.randn_like` is producing all-zero tensors on ROCm (gfx1151/Strix Halo), even when called on valid non-zero tensors. This is a critical bug that prevents training.

## Research Findings

### 1. No Direct Documentation
- **Finding**: There is no specific documented issue with `torch.randn_like` returning zeros on ROCm in official PyTorch documentation or GitHub issues.
- **Implication**: This may be a novel bug or a combination of known issues.

### 2. Related Issues Found

#### A. Random Number Generation Issues on GPU
- **torch.randperm**: Users have reported `torch.randperm(n)` generating all zero or negative values for large `n` on CUDA platforms.
- **torch.multinomial**: Unexpected behavior on float16 GPU input tensors.
- **torch.randn with bfloat16**: Inaccurate results when using `dtype=bfloat16` on CPU.

#### B. ROCm-Specific Issues
- **NaN Values in Loss**: Users report NaN values during training on ROCm systems.
- **Convolution Operations**: `nn.Conv2d` operations yield drastically different results on ROCm vs CPU.
  - **Workaround**: Setting `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0` has been suggested.
- **General Inconsistencies**: ROCm operations can behave unexpectedly compared to CPU.

### 3. Potential Root Causes

1. **GPU Random Number Generator Not Initialized**
   - The ROCm random number generator may not be properly initialized.
   - GPU state might be corrupted.

2. **Memory Allocation Issues**
   - GPU memory allocation might be returning zero-initialized memory.
   - HIP memory allocator could be broken on gfx1151.

3. **ROCm Driver/Version Issues**
   - Incompatible ROCm version with PyTorch.
   - Driver issues specific to gfx1151 (Strix Halo).

4. **Tensor Device State Corruption**
   - Tensors moved to GPU become corrupted (zeros).
   - This corruption affects all subsequent operations, including `torch.randn_like`.

## Recommended Solutions

### Immediate Workarounds

1. **Test Random Number Generation Separately**
   ```python
   # Test if GPU random number generation works at all
   test_tensor = torch.randn(100, device='cuda')
   print(f"Random tensor max: {test_tensor.abs().max().item()}")
   # If this is also zeros, the GPU RNG is broken
   ```

2. **Try Manual Random Number Generation**
   ```python
   # Instead of torch.randn_like(latents)
   # Generate on CPU and move to GPU
   noise = torch.randn_like(latents.cpu()).to(latents.device)
   ```

3. **Set ROCm Environment Variables**
   ```bash
   export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
   export HIP_DISABLE_IPC=1  # Already set
   ```

4. **Verify GPU State**
   ```python
   # Check if GPU is in a valid state
   torch.cuda.is_available()
   torch.cuda.get_device_name(0)
   # Try a simple operation
   test = torch.ones(10, device='cuda')
   print(test.sum().item())  # Should be 10.0
   ```

### Long-term Solutions

1. **Update ROCm and PyTorch**
   - Use latest ROCm nightly builds (already done).
   - Ensure PyTorch version matches ROCm version.

2. **Check for gfx1151-Specific Issues**
   - Search for Strix Halo specific ROCm issues.
   - Check if this architecture has known problems.

3. **Report to PyTorch/ROCm**
   - Create a minimal reproducible example.
   - Report to PyTorch GitHub issues.
   - Report to AMD ROCm support.

4. **Alternative: Train on CPU**
   - If GPU operations are fundamentally broken, CPU training might be the only option (very slow).

## Testing Checklist

- [ ] Test `torch.randn(shape, device='cuda')` directly
- [ ] Test `torch.randn_like(torch.ones(shape, device='cuda'))`
- [ ] Test simple GPU operations (add, multiply)
- [ ] Test GPU memory allocation
- [ ] Check ROCm driver version
- [ ] Check PyTorch ROCm version compatibility
- [ ] Test with different tensor dtypes (float32, float16)

## References

- PyTorch Discussion: [NaN in loss when training with torchtune on ROCm](https://discuss.pytorch.org/t/getting-nan-in-loss-when-training-with-torchtune-on-rocm-system/205063)
- PyTorch Discussion: [Conv2d returns different results on ROCm vs CPU](https://discuss.pytorch.org/t/conv-d-returns-drastically-different-results-on-rocm-vs-cpu/181331)
- GitHub Issue: [torch.randn inaccurate with bfloat16](https://github.com/pytorch/pytorch/issues/166853)
- Research Paper: [Random number generation issues on GPU](https://www.cs.cmu.edu/~fsaad/assets/papers/2025-SaadEtAl-PLDI.pdf)

## Conclusion

The issue with `torch.randn_like` producing zeros on ROCm appears to be related to:
1. GPU random number generator not working properly
2. GPU memory/tensor operations being corrupted
3. Potential ROCm driver or version incompatibility

Since this is not a widely documented issue, it may be:
- A bug specific to gfx1151 (Strix Halo)
- A combination of multiple ROCm issues
- A driver/version incompatibility

**Next Steps**: Test GPU random number generation directly and report findings to PyTorch/ROCm teams.


