#!/usr/bin/env python3
"""
Test script to diagnose ROCm GPU random number generator issues.

This script tests if GPU random number generation works correctly on ROCm,
specifically checking if torch.randn_like produces zeros (which would indicate
a severe ROCm bug).

Run with: python tests/test_rocm_gpu_random_generator.py
"""
import torch
import sys
import os

def test_gpu_random_generator():
    """Test GPU random number generator on ROCm."""
    print("=" * 80)
    print("ROCm GPU Random Number Generator Diagnostic Test")
    print("=" * 80)
    
    # Check if CUDA/ROCm is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm is not available!")
        return False
    
    device = torch.device("cuda:0")
    print(f"\nDevice: {device}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if hasattr(torch.version, 'hip'):
        print(f"ROCm Version: {torch.version.hip}")
    else:
        print("ROCm Version: unknown (not detected)")
    
    print(f"\nEnvironment Variables:")
    print(f"  HIP_DISABLE_IPC: {os.environ.get('HIP_DISABLE_IPC', 'not set')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")
    
    all_tests_passed = True
    
    # Test 1: Direct torch.randn on GPU
    print("\n" + "=" * 80)
    print("Test 1: torch.randn(shape, device='cuda')")
    print("=" * 80)
    try:
        test_tensor = torch.randn(1000, device=device)
        min_val = test_tensor.min().item()
        max_val = test_tensor.max().item()
        mean_val = test_tensor.mean().item()
        std_val = test_tensor.std().item()
        abs_max = test_tensor.abs().max().item()
        
        print(f"  Shape: {test_tensor.shape}")
        print(f"  Dtype: {test_tensor.dtype}")
        print(f"  Min: {min_val:.6f}")
        print(f"  Max: {max_val:.6f}")
        print(f"  Mean: {mean_val:.6f} (expected ~0.0)")
        print(f"  Std: {std_val:.6f} (expected ~1.0)")
        print(f"  Abs Max: {abs_max:.6f}")
        
        if abs_max < 1e-6:
            print("  ❌ FAILED: All values are zeros!")
            all_tests_passed = False
        elif abs(mean_val) > 0.1:
            print(f"  ⚠️  WARNING: Mean is far from 0.0 (expected ~0.0, got {mean_val:.6f})")
        elif abs(std_val - 1.0) > 0.1:
            print(f"  ⚠️  WARNING: Std is far from 1.0 (expected ~1.0, got {std_val:.6f})")
        else:
            print("  ✅ PASSED: Random numbers generated correctly")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 2: torch.randn_like on a non-zero tensor
    print("\n" + "=" * 80)
    print("Test 2: torch.randn_like(torch.ones(shape))")
    print("=" * 80)
    try:
        base_tensor = torch.ones(1000, device=device)
        test_tensor = torch.randn_like(base_tensor)
        min_val = test_tensor.min().item()
        max_val = test_tensor.max().item()
        mean_val = test_tensor.mean().item()
        std_val = test_tensor.std().item()
        abs_max = test_tensor.abs().max().item()
        
        print(f"  Base tensor: shape={base_tensor.shape}, sum={base_tensor.sum().item()}")
        print(f"  Generated tensor:")
        print(f"    Shape: {test_tensor.shape}")
        print(f"    Dtype: {test_tensor.dtype}")
        print(f"    Min: {min_val:.6f}")
        print(f"    Max: {max_val:.6f}")
        print(f"    Mean: {mean_val:.6f} (expected ~0.0)")
        print(f"    Std: {std_val:.6f} (expected ~1.0)")
        print(f"    Abs Max: {abs_max:.6f}")
        
        if abs_max < 1e-6:
            print("  ❌ FAILED: All values are zeros!")
            all_tests_passed = False
        elif abs(mean_val) > 0.1:
            print(f"  ⚠️  WARNING: Mean is far from 0.0 (expected ~0.0, got {mean_val:.6f})")
        elif abs(std_val - 1.0) > 0.1:
            print(f"  ⚠️  WARNING: Std is far from 1.0 (expected ~1.0, got {std_val:.6f})")
        else:
            print("  ✅ PASSED: Random numbers generated correctly")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 3: torch.randn_like on a zero tensor (the problematic case)
    print("\n" + "=" * 80)
    print("Test 3: torch.randn_like(torch.zeros(shape)) - PROBLEMATIC CASE")
    print("=" * 80)
    try:
        base_tensor = torch.zeros(1000, device=device)
        test_tensor = torch.randn_like(base_tensor)
        min_val = test_tensor.min().item()
        max_val = test_tensor.max().item()
        mean_val = test_tensor.mean().item()
        std_val = test_tensor.std().item()
        abs_max = test_tensor.abs().max().item()
        
        print(f"  Base tensor: shape={base_tensor.shape}, sum={base_tensor.sum().item()}")
        print(f"  Generated tensor:")
        print(f"    Shape: {test_tensor.shape}")
        print(f"    Dtype: {test_tensor.dtype}")
        print(f"    Min: {min_val:.6f}")
        print(f"    Max: {max_val:.6f}")
        print(f"    Mean: {mean_val:.6f} (expected ~0.0)")
        print(f"    Std: {std_val:.6f} (expected ~1.0)")
        print(f"    Abs Max: {abs_max:.6f}")
        
        if abs_max < 1e-6:
            print("  ❌ FAILED: All values are zeros!")
            print("  ⚠️  This is the problematic case - torch.randn_like on zeros produces zeros")
            all_tests_passed = False
        elif abs(mean_val) > 0.1:
            print(f"  ⚠️  WARNING: Mean is far from 0.0 (expected ~0.0, got {mean_val:.6f})")
        elif abs(std_val - 1.0) > 0.1:
            print(f"  ⚠️  WARNING: Std is far from 1.0 (expected ~1.0, got {std_val:.6f})")
        else:
            print("  ✅ PASSED: Random numbers generated correctly even from zero tensor")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 4: Simple GPU operations
    print("\n" + "=" * 80)
    print("Test 4: Simple GPU Operations (addition, multiplication)")
    print("=" * 80)
    try:
        a = torch.ones(100, device=device)
        b = torch.ones(100, device=device) * 2.0
        c = a + b
        d = a * 3.0
        
        sum_c = c.sum().item()
        sum_d = d.sum().item()
        
        print(f"  a = torch.ones(100): sum={a.sum().item()} (expected 100.0)")
        print(f"  b = torch.ones(100) * 2.0: sum={b.sum().item()} (expected 200.0)")
        print(f"  c = a + b: sum={sum_c} (expected 300.0)")
        print(f"  d = a * 3.0: sum={sum_d} (expected 300.0)")
        
        if abs(sum_c - 300.0) > 1e-3:
            print(f"  ❌ FAILED: Addition produces wrong result (expected 300.0, got {sum_c})")
            all_tests_passed = False
        elif abs(sum_d - 300.0) > 1e-3:
            print(f"  ❌ FAILED: Multiplication produces wrong result (expected 300.0, got {sum_d})")
            all_tests_passed = False
        else:
            print("  ✅ PASSED: Basic GPU operations work correctly")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 5: CPU to GPU transfer
    print("\n" + "=" * 80)
    print("Test 5: CPU to GPU Tensor Transfer")
    print("=" * 80)
    try:
        cpu_tensor = torch.randn(1000, device='cpu')
        cpu_max = cpu_tensor.abs().max().item()
        print(f"  CPU tensor: max={cpu_max:.6f}")
        
        gpu_tensor = cpu_tensor.to(device)
        gpu_max = gpu_tensor.abs().max().item()
        print(f"  GPU tensor: max={gpu_max:.6f}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Tensor became zeros when moved to GPU!")
            print("  ⚠️  This is the critical ROCm bug - CPU->GPU transfer corrupts tensors")
            all_tests_passed = False
        elif abs(gpu_max - cpu_max) > 1e-3:
            print(f"  ⚠️  WARNING: Values changed during transfer (CPU max={cpu_max:.6f}, GPU max={gpu_max:.6f})")
        else:
            print("  ✅ PASSED: CPU to GPU transfer works correctly")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 6: Multiple random number generations
    print("\n" + "=" * 80)
    print("Test 6: Multiple Random Number Generations (checking consistency)")
    print("=" * 80)
    try:
        results = []
        for i in range(5):
            tensor = torch.randn(100, device=device)
            max_val = tensor.abs().max().item()
            results.append(max_val)
            print(f"  Generation {i+1}: abs_max={max_val:.6f}")
        
        if all(r < 1e-6 for r in results):
            print("  ❌ FAILED: All generations produce zeros!")
            all_tests_passed = False
        elif len(set(results)) == 1 and results[0] < 1e-6:
            print("  ❌ FAILED: All generations produce the same zeros (not random)!")
            all_tests_passed = False
        else:
            print("  ✅ PASSED: Multiple generations produce different random values")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_tests_passed = False
    
    # Test 7: Simulate training scenario - large tensor with specific shape
    print("\n" + "=" * 80)
    print("Test 7: Training Scenario - Large Tensor (1, 16, 1, 64, 64)")
    print("=" * 80)
    try:
        # This matches the actual training tensor shape
        shape = (1, 16, 1, 64, 64)
        print(f"  Testing with training tensor shape: {shape}")
        
        # Create on CPU first (like in training)
        cpu_tensor = torch.randn(shape, device='cpu')
        cpu_max = cpu_tensor.abs().max().item()
        print(f"  CPU tensor: max={cpu_max:.6f}")
        
        # Move to GPU (this is where it might fail in training)
        gpu_tensor = cpu_tensor.to(device)
        gpu_max = gpu_tensor.abs().max().item()
        print(f"  GPU tensor after .to(): max={gpu_max:.6f}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Large tensor became zeros when moved to GPU!")
            all_tests_passed = False
        else:
            # Now test torch.randn_like on this tensor
            noise = torch.randn_like(gpu_tensor)
            noise_max = noise.abs().max().item()
            print(f"  torch.randn_like(gpu_tensor): max={noise_max:.6f}")
            
            if noise_max < 1e-6:
                print("  ❌ FAILED: torch.randn_like produces zeros on large tensor!")
                all_tests_passed = False
            else:
                print("  ✅ PASSED: Large tensor transfer and random generation work")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    # Test 8: Test with zero tensor that was moved from CPU (the actual problem case)
    print("\n" + "=" * 80)
    print("Test 8: Zero Tensor Moved from CPU (Actual Problem Case)")
    print("=" * 80)
    try:
        # Create zeros on CPU
        cpu_zeros = torch.zeros((1, 16, 1, 64, 64), device='cpu')
        print(f"  CPU zeros: max={cpu_zeros.abs().max().item():.6f}")
        
        # Move to GPU
        gpu_zeros = cpu_zeros.to(device)
        gpu_zeros_max = gpu_zeros.abs().max().item()
        print(f"  GPU zeros after .to(): max={gpu_zeros_max:.6f}")
        
        # Now test torch.randn_like on the zero tensor
        noise = torch.randn_like(gpu_zeros)
        noise_max = noise.abs().max().item()
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        
        print(f"  torch.randn_like(gpu_zeros):")
        print(f"    Max: {noise_max:.6f}")
        print(f"    Mean: {noise_mean:.6f} (expected ~0.0)")
        print(f"    Std: {noise_std:.6f} (expected ~1.0)")
        
        if noise_max < 1e-6:
            print("  ❌ FAILED: torch.randn_like produces zeros!")
            print("  ⚠️  This is the actual problem - even though Test 3 passed, this fails with moved tensor")
            all_tests_passed = False
        else:
            print("  ✅ PASSED: torch.randn_like works on zero tensor moved from CPU")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_tests_passed:
        print("✅ All tests PASSED - GPU random number generator appears to work correctly")
        print("\nNOTE: If training still shows zeros, the issue is likely:")
        print("  1. Tensors are already zeros before torch.randn_like is called")
        print("  2. Memory pressure or state during training")
        print("  3. Interaction with Accelerate or DataLoader")
        return True
    else:
        print("❌ Some tests FAILED - GPU random number generator has issues")
        print("\nIf torch.randn_like produces zeros, this indicates a severe ROCm bug.")
        print("Possible causes:")
        print("  1. ROCm driver/version incompatibility")
        print("  2. GPU memory corruption")
        print("  3. Random number generator not initialized")
        print("  4. gfx1151 (Strix Halo) specific issue")
        return False

if __name__ == "__main__":
    success = test_gpu_random_generator()
    sys.exit(0 if success else 1)

