#!/usr/bin/env python3
"""
Test script to isolate and reproduce ROCm tensor movement issues.

This script tests various tensor movement methods to identify which ones work
and which ones fail on ROCm (gfx1151/Strix Halo).
"""

import torch
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_simple_tensor_movement():
    """Test 1: Simple tensor movement from CPU to GPU"""
    print("\n=== Test 1: Simple Tensor Movement ===")
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    
    # Create a simple tensor on CPU
    tensor_cpu = torch.randn(4, 4, dtype=torch.float32)
    max_val_cpu = tensor_cpu.abs().max().item()
    print(f"CPU tensor max value: {max_val_cpu:.6e}")
    
    # Test Method 1: Direct .to()
    try:
        tensor_gpu1 = tensor_cpu.to(device, non_blocking=False)
        max_val_gpu1 = tensor_gpu1.abs().max().item()
        print(f"Method 1 (.to()): GPU max value: {max_val_gpu1:.6e} - {'PASS' if max_val_gpu1 > 1e-6 else 'FAIL'}")
        if max_val_gpu1 < 1e-6:
            return False
    except Exception as e:
        print(f"Method 1 (.to()): ERROR - {e}")
        return False
    
    # Test Method 2: copy_()
    try:
        tensor_gpu2 = torch.empty_like(tensor_cpu, device=device)
        tensor_gpu2.copy_(tensor_cpu, non_blocking=False)
        max_val_gpu2 = tensor_gpu2.abs().max().item()
        print(f"Method 2 (copy_()): GPU max value: {max_val_gpu2:.6e} - {'PASS' if max_val_gpu2 > 1e-6 else 'FAIL'}")
        if max_val_gpu2 < 1e-6:
            return False
    except Exception as e:
        print(f"Method 2 (copy_()): ERROR - {e}")
        return False
    
    # Test Method 3: numpy intermediate
    try:
        numpy_array = tensor_cpu.cpu().numpy()
        tensor_gpu3 = torch.from_numpy(numpy_array).to(device, non_blocking=False)
        max_val_gpu3 = tensor_gpu3.abs().max().item()
        print(f"Method 3 (numpy): GPU max value: {max_val_gpu3:.6e} - {'PASS' if max_val_gpu3 > 1e-6 else 'FAIL'}")
        if max_val_gpu3 < 1e-6:
            return False
    except Exception as e:
        print(f"Method 3 (numpy): ERROR - {e}")
        return False
    
    return True


def test_complex_tensor_movement():
    """Test 2: Complex tensor (matching our use case)"""
    print("\n=== Test 2: Complex Tensor Movement (16, 1, 64, 64) ===")
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    
    # Create tensor matching our latents shape
    tensor_cpu = torch.randn(16, 1, 64, 64, dtype=torch.float32)
    max_val_cpu = tensor_cpu.abs().max().item()
    print(f"CPU tensor max value: {max_val_cpu:.6e}")
    
    # Test Method 1: Direct .to()
    try:
        tensor_gpu1 = tensor_cpu.to(device, non_blocking=False)
        max_val_gpu1 = tensor_gpu1.abs().max().item()
        print(f"Method 1 (.to()): GPU max value: {max_val_gpu1:.6e} - {'PASS' if max_val_gpu1 > 1e-6 else 'FAIL'}")
        if max_val_gpu1 < 1e-6:
            return False
    except Exception as e:
        print(f"Method 1 (.to()): ERROR - {e}")
        return False
    
    # Test Method 2: copy_()
    try:
        tensor_gpu2 = torch.empty(tensor_cpu.shape, dtype=tensor_cpu.dtype, device=device)
        tensor_gpu2.copy_(tensor_cpu, non_blocking=False)
        max_val_gpu2 = tensor_gpu2.abs().max().item()
        print(f"Method 2 (copy_()): GPU max value: {max_val_gpu2:.6e} - {'PASS' if max_val_gpu2 > 1e-6 else 'FAIL'}")
        if max_val_gpu2 < 1e-6:
            return False
    except Exception as e:
        print(f"Method 2 (copy_()): ERROR - {e}")
        return False
    
    return True


def test_batch_dict_movement():
    """Test 3: Batch dict with multiple tensors (matching our use case)"""
    print("\n=== Test 3: Batch Dict Movement ===")
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    
    # Create batch dict matching our structure
    batch = {
        "latents": torch.randn(1, 16, 1, 64, 64, dtype=torch.float32),
        "latents_image": torch.randn(1, 20, 1, 64, 64, dtype=torch.float32),
        "t5": [torch.randn(1, 77, 4096, dtype=torch.float32)],  # List of tensors
        "timesteps": torch.randint(0, 1000, (1,), dtype=torch.long),
    }
    
    # Check CPU values
    max_latents_cpu = batch["latents"].abs().max().item()
    max_image_cpu = batch["latents_image"].abs().max().item()
    print(f"CPU latents max: {max_latents_cpu:.6e}")
    print(f"CPU latents_image max: {max_image_cpu:.6e}")
    
    # Move to GPU
    try:
        batch["latents"] = batch["latents"].to(device, non_blocking=False)
        batch["latents_image"] = batch["latents_image"].to(device, non_blocking=False)
        if isinstance(batch["t5"], list):
            batch["t5"] = [t.to(device, non_blocking=False) for t in batch["t5"]]
        batch["timesteps"] = batch["timesteps"].to(device, non_blocking=False)
        
        max_latents_gpu = batch["latents"].abs().max().item()
        max_image_gpu = batch["latents_image"].abs().max().item()
        
        print(f"GPU latents max: {max_latents_gpu:.6e} - {'PASS' if max_latents_gpu > 1e-6 else 'FAIL'}")
        print(f"GPU latents_image max: {max_image_gpu:.6e} - {'PASS' if max_image_gpu > 1e-6 else 'FAIL'}")
        
        if max_latents_gpu < 1e-6 or max_image_gpu < 1e-6:
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True


def test_with_accelerate():
    """Test 4: Test with Accelerate (if available)"""
    print("\n=== Test 4: Accelerate DataLoader Test ===")
    
    try:
        from accelerate import Accelerator
    except ImportError:
        print("SKIP: accelerate not available")
        return None
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm not available")
        return None
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                "latents": torch.randn(16, 1, 64, 64, dtype=torch.float32),
                "data": torch.randn(4, 4, dtype=torch.float32),
            }
    
    dataset = SimpleDataset(5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
    )
    
    # Test without device placement
    try:
        dataloader = accelerator.prepare(dataloader, device_placement=[False])
        
        batch = next(iter(dataloader))
        max_latents_cpu = batch["latents"].abs().max().item()
        print(f"After Accelerate prepare (device_placement=False):")
        print(f"  Device: {batch['latents'].device}")
        print(f"  CPU max: {max_latents_cpu:.6e}")
        
        # Manually move to GPU
        batch["latents"] = batch["latents"].to(device, non_blocking=False)
        max_latents_gpu = batch["latents"].abs().max().item()
        print(f"  GPU max: {max_latents_gpu:.6e} - {'PASS' if max_latents_gpu > 1e-6 else 'FAIL'}")
        
        if max_latents_gpu < 1e-6:
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("ROCm Tensor Movement Test Suite")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        if hasattr(torch.version, 'hip'):
            print(f"ROCm/HIP version: {torch.version.hip}")
    else:
        print("WARNING: CUDA/ROCm not available - tests will be skipped")
    
    results = {}
    
    results["simple"] = test_simple_tensor_movement()
    results["complex"] = test_complex_tensor_movement()
    results["batch_dict"] = test_batch_dict_movement()
    results["accelerate"] = test_with_accelerate()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"{test_name:15s}: {status}")
    
    # Determine overall result
    failed_tests = [name for name, result in results.items() if result is False]
    if failed_tests:
        print(f"\nFAILED TESTS: {', '.join(failed_tests)}")
        print("This indicates a ROCm bug with tensor movement.")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

