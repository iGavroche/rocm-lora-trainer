# ROCm Tensor Corruption Root Cause Research

## Problem Statement

Tensors loaded correctly on CPU become zeros when moved to GPU via Accelerate's DataLoader on ROCm (gfx1151/Strix Halo). All tensor movement methods fail (`.to()`, `copy_()`, numpy intermediate, etc.). This suggests the issue may not be tensor movement itself but something deeper in the stack.

**Environment:**
- GPU: AMD Radeon 8060S (gfx1151/Strix Halo)
- ROCm: 7.11.0 (nightly build from Nov 22, 2024)
- PyTorch: 2.10.0a0+rocm7.11.0a20251122
- OS: Windows
- Accelerate: Latest version

## Research Findings

### 1. PyTorch Accelerate + ROCm Integration

#### Documentation Review
- **HuggingFace Accelerate Documentation**: No specific ROCm-related documentation found. Device placement is documented but doesn't mention ROCm-specific issues.
- **PyTorch ROCm Documentation**: Limited documentation on DataLoader behavior with ROCm. No specific guidance on device placement.

#### GitHub Issues
**huggingface/accelerate:**
- No specific issues found for "ROCm tensor zeros" or "ROCm device placement"
- Limited ROCm-related issues overall, suggesting Accelerate may not be widely tested with ROCm

**pytorch/pytorch:**
- Issue #155720: Segmentation fault during HIP Graph capture on AMD GPUs (CUDA works)
- Issue #108404: Second GPU doesn't work correctly with multiple AMD GPUs using ROCm
- Issue #120775: System crashes and resets when performing computations on AMD GPUs with ROCm
- General pattern: ROCm has more stability issues than CUDA

#### Code Analysis
- Accelerate's `device_placement` parameter should prevent automatic tensor movement
- However, the implementation may have bugs or edge cases with ROCm
- Need to verify if `device_placement=[True, True, False, True]` actually works as documented

### 2. DataLoader + Multiprocessing on Windows + ROCm

#### Documentation Review
- **PyTorch DataLoader**: Windows multiprocessing uses `spawn` instead of `fork`, which can cause issues
- **ROCm on Windows**: Limited support, many features are Linux-only
- **num_workers=0**: Should avoid multiprocessing issues, but may not solve the root cause

#### Community Discussions
- **Reddit**: Limited discussions found. Most ROCm users are on Linux.
- **Stack Overflow**: Some reports of tensor corruption, but solutions vary (IOMMU settings, etc.)

#### Key Findings
- Windows + ROCm + multiprocessing is a known problematic combination
- Setting `num_workers=0` is a common workaround, but doesn't address root cause
- Some users report success with DirectML as an alternative backend

### 3. Memory Management and Allocator Issues

#### Documentation Review
- **PYTORCH_CUDA_ALLOC_CONF**: `expandable_segments:True` shows a warning on ROCm:
  ```
  UserWarning: expandable_segments not supported on this platform
  ```
- This suggests ROCm doesn't fully support all PyTorch memory allocator features

#### GitHub Issues
- Limited specific issues about ROCm memory allocator corruption
- However, general ROCm stability issues may be related to memory management

#### Hypothesis
- The memory allocator warning suggests ROCm may have incomplete memory management support
- This could lead to memory corruption when tensors are moved between CPU and GPU
- The corruption manifests as zeros, which could indicate uninitialized or corrupted memory

### 4. Accelerate's Device Placement Logic

#### Code Analysis Needed
- Review `accelerate/src/accelerate/data_loader.py` to understand device placement
- Check if there's a bug where tensors are still moved despite `device_placement=False`
- Verify how Accelerate wraps DataLoader and intercepts batches

#### Hypothesis
- **Most Likely Culprit**: Accelerate's device placement logic may not work correctly with ROCm
- Even with `device_placement=[True, True, False, True]`, Accelerate might still be moving tensors
- The movement happens in Accelerate's wrapper code, not in our manual movement code

### 5. Alternative Approaches That Work

#### GitHub Code Search
- **Kohya-ss/sd-scripts**: Primarily CUDA-focused, limited ROCm support
- **Axolotl**: Some ROCm support, but primarily Linux-based
- **General Pattern**: Most successful ROCm training projects avoid Accelerate or use custom DataLoader wrappers

#### Key Insight
- Projects that work with ROCm often:
  1. Don't use Accelerate's DataLoader wrapper
  2. Manually handle device placement
  3. Use simpler data loading patterns
  4. Run on Linux (not Windows)

### 6. gfx1151 (Strix Halo) Specific Issues

#### GitHub Issues
- Limited specific issues for gfx1151
- Most ROCm issues are architecture-agnostic or focus on older architectures

#### Reddit Discussions
- Strix Halo is relatively new, limited community experience
- Most discussions focus on Linux setup, not Windows

### 7. Hypothesis Testing Results

#### Key Questions Answered:

1. **Does the issue occur with Accelerate's DataLoader wrapper?**
   - YES - Tensors become zeros when Accelerate moves them
   - UNKNOWN - Need to test with raw PyTorch DataLoader

2. **Is it specific to how Accelerate moves tensors?**
   - LIKELY YES - Manual movement also fails, but this could be a red herring
   - The real issue might be that Accelerate has already corrupted the tensor somehow

3. **Does it happen with simple tensors?**
   - UNKNOWN - Need to test with isolated simple tensors
   - Test script created to verify this

4. **Is it related to the batch structure?**
   - POSSIBLY - Complex batch dicts might trigger different code paths
   - Need to test with simple vs complex batches

5. **Does it happen immediately on first batch?**
   - YES - Happens on first batch, not after some batches
   - This suggests it's not a memory leak or accumulation issue

## Root Cause Hypothesis

### Primary Hypothesis: Accelerate + ROCm Incompatibility

**Most Likely Root Cause:**
Accelerate's DataLoader wrapper and device placement logic are not compatible with ROCm on Windows. Even when `device_placement=False` is set, Accelerate may be:
1. Still moving tensors through a code path that corrupts them
2. Using CUDA-specific code paths that don't work with ROCm
3. Intercepting batches in a way that corrupts tensor memory

**Evidence:**
- Tensors are valid on CPU (verified in dataset loading)
- Tensors become zeros when Accelerate handles them
- All manual movement methods also fail (but this might be because the tensor is already corrupted)
- No similar issues reported with CUDA
- Limited ROCm testing in Accelerate codebase

### Secondary Hypothesis: ROCm Memory Allocator Bug

**Alternative Root Cause:**
ROCm's memory allocator has a bug where moving tensors from CPU to GPU corrupts memory, especially with certain tensor shapes or batch structures.

**Evidence:**
- `expandable_segments` warning suggests incomplete ROCm support
- General ROCm stability issues reported
- Windows + ROCm is a less-tested combination

## Recommended Solutions

### Solution 1: Bypass Accelerate's DataLoader Wrapper (HIGHEST PRIORITY)

**Approach:**
- Don't use `accelerator.prepare()` on the DataLoader
- Use raw PyTorch DataLoader
- Manually handle device placement in training loop
- Still use Accelerate for model/optimizer/scheduler, but not DataLoader

**Implementation:**
```python
# Instead of:
train_dataloader = accelerator.prepare(train_dataloader, device_placement=[False])

# Use:
# Don't prepare dataloader, use raw PyTorch DataLoader
# Then in training loop, manually move tensors
```

**Pros:**
- Bypasses Accelerate's potentially buggy DataLoader code
- Full control over tensor movement
- Can test if issue is Accelerate-specific

**Cons:**
- Lose some Accelerate features (automatic device placement, etc.)
- Need to manually handle device placement

### Solution 2: Use Alternative Backend (MEDIUM PRIORITY)

**Approach:**
- Try DirectML as an alternative to ROCm on Windows
- Or use WSL2 with Linux ROCm (better support)

**Pros:**
- DirectML has better Windows support
- WSL2 + Linux ROCm is more stable

**Cons:**
- DirectML may have different limitations
- WSL2 adds complexity

### Solution 3: Report Bug and Wait for Fix (LOW PRIORITY)

**Approach:**
- Report detailed bug to both Accelerate and PyTorch/ROCm teams
- Provide minimal reproducible example
- Wait for upstream fix

**Pros:**
- Helps the community
- May get official fix

**Cons:**
- May take months/years
- Doesn't solve immediate problem

## Test Script

Created `tests/test_rocm_tensor_movement.py` to isolate the issue:
- Test 1: Simple tensor movement (4x4)
- Test 2: Complex tensor movement (16x1x64x64, matching our use case)
- Test 3: Batch dict movement (matching our batch structure)
- Test 4: Accelerate DataLoader test (with device_placement=False)

**Usage:**
```bash
python tests/test_rocm_tensor_movement.py
```

This will help identify:
- Which tensor movement methods work/fail
- If the issue is Accelerate-specific
- If the issue is shape/structure-specific

## Next Steps

1. **Run the test script** to isolate the issue
2. **Try Solution 1** (bypass Accelerate DataLoader) - highest priority
3. **If Solution 1 works**, report bug to Accelerate with findings
4. **If Solution 1 doesn't work**, investigate ROCm memory allocator deeper
5. **Consider Solution 2** (alternative backend) if Solution 1 fails

## References

- [PyTorch ROCm Issues](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+rocm)
- [Accelerate GitHub](https://github.com/huggingface/accelerate)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch Forums - ROCm Discussions](https://discuss.pytorch.org/search?q=rocm)

## Conclusion

The most likely root cause is an incompatibility between Accelerate's DataLoader wrapper and ROCm on Windows. The recommended solution is to bypass Accelerate's DataLoader wrapper and use raw PyTorch DataLoader with manual device placement. This will help isolate whether the issue is Accelerate-specific or a deeper ROCm problem.

