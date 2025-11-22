#!/usr/bin/env python3
"""
Isolated test to reproduce and diagnose the ROCm tensor transfer issue.

This test:
1. Loads a tensor from cache (same as training)
2. Tries all transfer methods
3. Provides detailed diagnostics
4. Tests if simple transfers work (to isolate the issue)

Run with: python tests/test_rocm_tensor_transfer_issue.py
"""
import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_rocm_tensor_transfer():
    """Test tensor transfer to identify the exact issue"""
    print("=" * 80)
    print("ROCm Tensor Transfer Issue Diagnostic Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if hasattr(torch.version, 'hip'):
        print(f"ROCm: {torch.version.hip}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Test 1: Can we create and transfer simple tensors?
    print("=" * 80)
    print("TEST 1: Simple Tensor Transfer")
    print("=" * 80)
    
    simple_tensor = torch.randn(100, device='cpu')
    simple_max = simple_tensor.abs().max().item()
    print(f"Created simple CPU tensor: max={simple_max:.6e}")
    
    simple_gpu = simple_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    simple_gpu_max = simple_gpu.abs().max().item()
    print(f"Transferred to GPU: max={simple_gpu_max:.6e}")
    
    if simple_gpu_max < 1e-6 and simple_max > 1e-6:
        print("❌ FAILED: Simple tensor transfer produces zeros!")
        print("This indicates a fundamental ROCm bug - even simple transfers fail.")
        return False
    else:
        print("✅ PASSED: Simple tensor transfer works")
    
    # Test 2: Can we transfer tensors with the same shape as training?
    print("\n" + "=" * 80)
    print("TEST 2: Training-Size Tensor Transfer")
    print("=" * 80)
    
    training_shape = (1, 16, 1, 64, 64)
    training_tensor = torch.randn(training_shape, device='cpu')
    training_max = training_tensor.abs().max().item()
    print(f"Created training-size CPU tensor: shape={training_shape}, max={training_max:.6e}")
    
    # Test contiguous
    is_contig = training_tensor.is_contiguous()
    print(f"  Is contiguous: {is_contig}")
    if not is_contig:
        training_tensor = training_tensor.contiguous()
        print("  Made contiguous")
    
    # Test clone
    training_clone = training_tensor.clone()
    clone_max = training_clone.abs().max().item()
    print(f"  After clone: max={clone_max:.6e}")
    
    if clone_max < 1e-6:
        print("❌ FAILED: Clone produces zeros!")
        return False
    
    # Try transfer
    training_gpu = training_clone.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    training_gpu_max = training_gpu.abs().max().item()
    print(f"Transferred to GPU: max={training_gpu_max:.6e}")
    
    if training_gpu_max < 1e-6 and training_max > 1e-6:
        print("❌ FAILED: Training-size tensor transfer produces zeros!")
        print("This is the exact issue - tensors with this shape fail to transfer.")
        return False
    else:
        print("✅ PASSED: Training-size tensor transfer works")
    
    # Test 3: Load actual cache file tensor
    print("\n" + "=" * 80)
    print("TEST 3: Actual Cache File Tensor Transfer")
    print("=" * 80)
    
    cache_dir = Path("myface")
    cache_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
    
    if not cache_files:
        print("⚠️  SKIPPED: No cache files found in myface/")
        return True
    
    cache_file = cache_files[0]
    print(f"Loading from: {cache_file}")
    
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
    
    with MemoryEfficientSafeOpen(str(cache_file), disable_numpy_memmap=False) as f:
        keys = list(f.keys())
        print(f"Keys in cache: {keys}")
        
        # Find latents key
        latents_key = None
        for key in keys:
            if "latents_" in key and "float32" in key:
                latents_key = key
                break
        
        if not latents_key:
            print("⚠️  SKIPPED: No latents key found")
            return True
        
        print(f"Loading key: {latents_key}")
        
        # Load to CPU (same as training)
        cache_tensor = f.get_tensor(latents_key, device=torch.device("cpu"), dtype=torch.float32)
        cache_max = cache_tensor.abs().max().item()
        cache_mean = cache_tensor.abs().mean().item()
        print(f"Loaded to CPU: shape={cache_tensor.shape}, dtype={cache_tensor.dtype}, max={cache_max:.6e}, mean={cache_mean:.6e}")
        
        if cache_max < 1e-6:
            print("❌ FAILED: Cache tensor is zeros on CPU!")
            return False
        
        # Check if it's contiguous
        is_contig = cache_tensor.is_contiguous()
        print(f"  Is contiguous: {is_contig}")
        if not is_contig:
            cache_tensor = cache_tensor.contiguous()
            print("  Made contiguous")
        
        # Clone it
        cache_clone = cache_tensor.clone()
        clone_max = cache_clone.abs().max().item()
        print(f"  After clone: max={clone_max:.6e}")
        
        if clone_max < 1e-6:
            print("❌ FAILED: Clone produces zeros!")
            return False
        
        # Now try all transfer methods
        print("\nTesting all transfer methods:")
        print("-" * 80)
        
        methods = [
            ("Method 1: Direct .to()", lambda t: t.to(device, non_blocking=False)),
            ("Method 2: Pinned memory + .to()", lambda t: t.pin_memory().to(device, non_blocking=True)),
            ("Method 3: copy_()", lambda t: torch.empty(t.shape, dtype=t.dtype, device=device).copy_(t, non_blocking=False)),
            ("Method 4: Chunked copy_()", lambda t: chunked_copy(t, device)),
            ("Method 5: numpy + torch.tensor", lambda t: torch.tensor(t.cpu().numpy(), dtype=t.dtype, device=device)),
            ("Method 6: numpy + torch.as_tensor", lambda t: torch.as_tensor(t.cpu().numpy(), device=device)),
        ]
        
        success_count = 0
        for method_name, method_func in methods:
            try:
                print(f"\n{method_name}...")
                result = method_func(cache_clone)
                torch.cuda.synchronize(device)
                result_max = result.abs().max().item()
                print(f"  Result: max={result_max:.6e}")
                
                if result_max < 1e-6 and cache_max > 1e-6:
                    print(f"  ❌ FAILED: Produces zeros")
                else:
                    print(f"  ✅ PASSED: Transfer successful")
                    success_count += 1
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
        
        print("\n" + "-" * 80)
        print(f"Summary: {success_count}/{len(methods)} methods succeeded")
        
        if success_count == 0:
            print("\n❌ ALL METHODS FAILED")
            print("This confirms the ROCm bug - no transfer method works for this tensor.")
            return False
        elif success_count < len(methods):
            print(f"\n⚠️  PARTIAL SUCCESS: {len(methods) - success_count} methods failed")
            print("Some methods work, but not all. This suggests a method-specific issue.")
            return True
        else:
            print("\n✅ ALL METHODS SUCCEEDED")
            print("All transfer methods work in isolation. The issue must be training-specific.")
            return True

def chunked_copy(tensor, device):
    """Chunked copy method"""
    tensor_gpu = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
    flat_cpu = tensor.flatten()
    flat_gpu = tensor_gpu.flatten()
    
    chunk_size = min(1024 * 1024, flat_cpu.numel())
    for i in range(0, flat_cpu.numel(), chunk_size):
        end_idx = min(i + chunk_size, flat_cpu.numel())
        flat_gpu[i:end_idx].copy_(flat_cpu[i:end_idx], non_blocking=False)
    
    return tensor_gpu

def test_after_model_loading():
    """Test if the issue occurs after model loading (simulating training context)"""
    print("\n" + "=" * 80)
    print("TEST 4: Transfer After Model Memory Allocation")
    print("=" * 80)
    
    device = torch.device("cuda:0")
    
    # Simulate model loading by allocating large tensors
    print("Allocating large tensors to simulate model memory usage...")
    model_tensors = []
    for i in range(10):
        tensor = torch.randn((1000, 1000), device=device)
        model_tensors.append(tensor)
    
    print(f"Allocated {len(model_tensors)} tensors on GPU")
    
    # Now try transferring a tensor
    test_tensor = torch.randn((1, 16, 1, 64, 64), device='cpu')
    test_max = test_tensor.abs().max().item()
    print(f"Created test tensor: max={test_max:.6e}")
    
    test_gpu = test_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    test_gpu_max = test_gpu.abs().max().item()
    print(f"Transferred to GPU: max={test_gpu_max:.6e}")
    
    if test_gpu_max < 1e-6 and test_max > 1e-6:
        print("❌ FAILED: Transfer fails after model memory allocation!")
        print("This suggests the issue is related to GPU memory state.")
        return False
    else:
        print("✅ PASSED: Transfer works even after model memory allocation")
        return True

def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 80)
    print("ROCm Tensor Transfer Diagnostic Suite")
    print("=" * 80)
    print()
    
    results = {}
    results["test1"] = test_rocm_tensor_transfer()
    results["test4"] = test_after_model_loading()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("Tensors transfer correctly in isolation.")
        print("The issue must be specific to the training context.")
        print("\nPossible causes:")
        print("  1. Accelerate library interference")
        print("  2. DataLoader wrapper issues")
        print("  3. Specific tensor state in training loop")
        print("  4. Memory state during training")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("This confirms a ROCm bug with tensor transfers.")
        print("\nRecommendations:")
        print("  1. Update ROCm drivers to latest version")
        print("  2. Try a different ROCm version")
        print("  3. Report this bug to AMD/ROCm with this diagnostic output")
        print("  4. Consider using a different GPU or platform")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


