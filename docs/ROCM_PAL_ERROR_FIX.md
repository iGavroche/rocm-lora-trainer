# ROCm PAL Error -28 Fix

## Problem

PAL (Platform Abstraction Layer) error -28: "PAL failed to finalize a command buffer! result: -28"

This error indicates command buffer pool exhaustion in the ROCm driver, typically caused by:
- Too many concurrent GPU operations without proper synchronization
- Command buffer pool being exhausted
- Resource allocation failures due to memory pressure

## Solutions Implemented

### 1. ROCm Environment Variables (run_training.ps1)

Added environment variables to help with command buffer management:

- `HIP_FORCE_DEV_KERNARG=1`: Forces device kernel arguments, can help with command buffer issues
- `HSA_AMD_SDMA_ENABLE=1`: Enables SDMA for better memory transfers
- `HIP_SERIALIZE_KERNEL=0`: Set to "1" if errors persist (serializes kernel launches, slower but more stable)

### 2. Training Loop Synchronization (hv_train_network.py)

Added synchronization points to prevent command buffer overflow:

- **After backward pass**: Synchronizes GPU after `accelerator.backward(loss)` to ensure gradients are computed before optimizer step
- **After optimizer step**: Synchronizes GPU after `optimizer.step()` to ensure all operations complete before next iteration
- **Periodic memory cleanup**: Calls `torch.cuda.empty_cache()` every 10 steps to free up GPU memory and resources

## Memory Allocation Failures

If you see errors like:
- "Failed PAL memory allocation!"
- "Video memory allocation failed!"
- "HIP out of memory" (even when there's plenty of free memory)

This indicates memory fragmentation caused by PAL errors. The code now:

1. **Moves model to CPU for dtype conversion**: When converting fp16→bf16 (or vice versa), the model is moved to CPU first, converted, then moved back to GPU. This avoids allocating large contiguous blocks on a fragmented GPU.

2. **Aggressive memory cleanup**: Before and after model loading, the code performs aggressive memory cleanup with synchronization.

3. **Chunked operations**: Memory-intensive operations are done in smaller chunks to avoid large allocations.

## If Errors Persist

### Option 1: Enable Kernel Serialization

If PAL errors continue, enable kernel serialization (slower but more stable):

```powershell
# In run_training.ps1, change:
$env:HIP_SERIALIZE_KERNEL="0"
# To:
$env:HIP_SERIALIZE_KERNEL="1"
```

### Option 2: Enable Blocking Launches

Enable blocking launches to ensure operations complete before next ones start:

```powershell
# In run_training.ps1, change:
$env:HIP_LAUNCH_BLOCKING="0"
# To:
$env:HIP_LAUNCH_BLOCKING="1"
```

**Note**: This will significantly slow down training but may resolve command buffer issues.

### Option 3: Reduce Batch Size

If memory pressure is causing resource exhaustion, reduce batch size or other memory-intensive settings.

### Option 4: Update ROCm Drivers

Ensure you're using the latest ROCm drivers for your GPU. The nightly builds are updated regularly with bug fixes.

### Option 5: Load Model to CPU First

If memory allocation failures persist during model loading, you can modify the loading device to "cpu" first, then move to GPU after conversion. This is now done automatically for dtype conversions.

## Monitoring

Watch for:
- Frequency of PAL errors (should decrease with fixes)
- Training speed (may be slightly slower with synchronization)
- GPU memory usage (should be more stable with periodic cleanup)

## Related Issues

- This is a known ROCm driver-level issue on Windows
- Command buffer pool size is limited by the driver
- Synchronization helps prevent overflow but may impact performance

