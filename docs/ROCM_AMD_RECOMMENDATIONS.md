# AMD ROCm Official Recommendations for LoRA Training

Based on AMD's official documentation and best practices, here are the **proven** recommendations for training LoRAs with ROCm:

## AMD-Recommended Libraries (Official)

### 1. **Torchtune** (AMD's Official Fine-Tuning Library)
- **Purpose**: AMD's PyTorch-based fine-tuning library optimized for ROCm
- **Features**: 
  - Supports LoRA fine-tuning
  - Optimized for AMD GPUs
  - Built-in ROCm optimizations
- **Availability**: AMD Developer Central
- **Status**: ✅ **AMD's primary recommendation for fine-tuning**

### 2. **Hugging Face PEFT** (Parameter-Efficient Fine-Tuning)
- **Purpose**: Parameter-Efficient Fine-Tuning library
- **Features**:
  - ROCm-compatible
  - Supports LoRA, QLoRA
  - Integrates with Transformers library
- **Status**: ✅ **AMD-recommended and ROCm-compatible**

### 3. **Flash Attention** (Hardware-Aware Attention)
- **Purpose**: Optimized attention mechanism for AMD GPUs
- **Features**:
  - Reduces memory usage
  - Improves training speed
  - Hardware-aware optimizations
- **Status**: ✅ **AMD-recommended optimization**

### 4. **AITER** (AI Tensor Engine for ROCm)
- **Purpose**: Centralized repository of high-performance AI operators for AMD GPUs
- **Features**:
  - Optimized AI operations
  - AMD GPU-specific optimizations
- **Status**: ✅ **AMD-recommended for AI workloads**

### 5. **TransferBench**
- **Purpose**: Benchmark simultaneous data transfers between CPUs and GPUs
- **Features**:
  - Identifies data transfer bottlenecks
  - Optimizes transfer performance
- **Status**: ✅ **AMD-provided tool for transfer optimization**

## ROCm Environment Variables (Required)

### Windows + ROCm (Required):
- `HIP_DISABLE_IPC=1`: **Required** - Disables Inter-Process Communication (mandatory for Windows multiprocessing with ROCm)

### GPU Architecture Override (Required for Strix Halo):
- `HSA_OVERRIDE_GFX_VERSION=11.5.1`: **Required** - Forces gfx1151 architecture for Strix Halo (RDNA3.5)

## Performance Optimizations (AMD-Recommended)

### 1. **Activation Checkpointing**
- **Purpose**: Reduces memory usage during training
- **How it works**: Recomputes activations during backward pass instead of storing them
- **Trade-off**: Slower training but uses less memory
- **Status**: ✅ **AMD-recommended for large models**

### 2. **GEMM Optimization with ROCm's Transformer Engine**
- **Purpose**: Optimizes matrix operations (GEMM) for transformers
- **How it works**: 
  - Integrates with hipBLASLt
  - Performs kernel search for optimal performance
  - Hardware-aware kernel selection
- **Status**: ✅ **AMD-recommended for transformer training**

### 3. **Non-GEMM Kernel Optimization**
- **Purpose**: Optimizes memory-bound operations
- **Operations**: RMSNorm, SwiGLU, element-wise operations
- **How it works**: Custom HIP kernels can improve performance
- **Status**: ✅ **AMD-recommended for memory-bound operations**

## DataLoader Configuration (Windows + ROCm)

**AMD's recommendation**: Use single-process data loading on Windows
- `num_workers=0`: **Required** - Avoids multiprocessing issues on Windows + ROCm
- `persistent_workers=False`: Not available with `num_workers=0`

## Memory Management (PyTorch Best Practices)

### Contiguous Tensors
- **Why**: Non-contiguous tensors can cause issues with GPU operations
- **How**: Use `.contiguous()` after operations like `torch.stack()`
- **Status**: ✅ **Standard PyTorch best practice**

### Pinned Memory
- **Why**: Improves CPU→GPU transfer speed
- **How**: Call `.pin_memory()` before `.to(device)`
- **Status**: ✅ **Standard PyTorch best practice**

## Known Limitations (Not Workarounds - These Are Limitations)

### PyTorch Memory Allocator
- **Issue**: `expandable_segments` not supported on ROCm
- **Impact**: PyTorch shows warning: `expandable_segments not supported on this platform`
- **Status**: ⚠️ **Known ROCm limitation** - No workaround available

### Mixed Precision on ROCm
- **Issue**: bfloat16 conversion can produce zeros on some ROCm versions
- **AMD Recommendation**: Use float32 for critical operations (VAE encoding)
- **Status**: ⚠️ **Known ROCm limitation** - Use float32 as workaround

## What We've Tried That Didn't Work

The following approaches were attempted but **did not solve** the tensor corruption issue:

### ❌ Not Recommended (Didn't Work)
1. **Accelerate's `device_placement` parameter**: Setting `device_placement=[True, True, False, True]` did not prevent tensor corruption
2. **Direct GPU loading from cache**: Loading tensors directly to GPU still produced zeros
3. **Multiple transfer methods**: 8 different CPU→GPU transfer methods all failed in training context
4. **Accelerate DataLoader wrapper bypass**: While this avoids some issues, it doesn't solve the root cause
5. **Custom workarounds**: Extensive workarounds in our codebase are not AMD-recommended solutions

### ⚠️ Current Workarounds (Not AMD-Recommended)
- **Bypassing Accelerate's DataLoader**: Using raw PyTorch DataLoader with manual device placement
- **Multiple transfer method fallbacks**: Trying 8 different methods to move tensors to GPU
- **Extensive logging and diagnostics**: Helps identify issues but doesn't fix them

**Note**: These workarounds are project-specific and not part of AMD's official recommendations.

## AMD's Official Approach

AMD recommends using **Torchtune** or **Hugging Face PEFT** for LoRA training, which:
- Are optimized for ROCm
- Handle device placement correctly
- Avoid the issues we're experiencing with Accelerate
- Are actively maintained and tested with ROCm

## ROCm Installation and Updates

### Stay Updated
- **Importance**: Regularly update to the latest ROCm version for newest features and bug fixes
- **Example**: ROCm 7.0 introduces enhancements like unified Triton 3.3 kernels and DeepEP inference engine
- **Status**: ✅ **AMD-recommended practice**

### Verify Installation
- **Check**: Ensure ROCm installation is correctly set up
- **Troubleshoot**: AMD's documentation provides troubleshooting steps for common issues
- **Status**: ✅ **AMD-recommended practice**

## References

- [AMD ROCm Fine-Tuning Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/overview.html)
- [AMD Torchtune Documentation](https://rocm.blogs.amd.com/artificial-intelligence/torchtune/README.html)
- [AMD MLPerf Training Optimizations](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.0/README.html)
- [AMD AITER Documentation](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)
- [AMD TransferBench](https://rocm.docs.amd.com/_/downloads/TransferBench/en/latest/pdf/)
- [AMD GEMM Optimization](https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html)

## Summary

**What Works (AMD-Recommended)**:
- ✅ Torchtune (AMD's official library)
- ✅ Hugging Face PEFT
- ✅ Flash Attention
- ✅ AITER (AI Tensor Engine)
- ✅ TransferBench (for transfer optimization)
- ✅ Activation checkpointing
- ✅ GEMM optimization with ROCm Transformer Engine
- ✅ `HIP_DISABLE_IPC=1` (required for Windows)
- ✅ `HSA_OVERRIDE_GFX_VERSION=11.5.1` (required for Strix Halo)
- ✅ `num_workers=0` (required for Windows + ROCm)
- ✅ Regular ROCm updates

**What Doesn't Work (Our Attempts)**:
- ❌ Accelerate's device placement workarounds
- ❌ Direct GPU loading from cache
- ❌ Multiple transfer method fallbacks
- ❌ Custom DataLoader workarounds

**Recommendation**: Consider migrating to Torchtune or Hugging Face PEFT for better ROCm compatibility and official AMD support.
