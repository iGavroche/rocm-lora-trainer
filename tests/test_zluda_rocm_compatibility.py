"""Test ZLUDA compatibility with Windows + ROCm.

ZLUDA is a compatibility layer that translates CUDA calls to run on AMD GPUs.
This could potentially bypass the ROCm bug by using a different code path.

This test verifies:
1. If ZLUDA can be installed/configured
2. If ZLUDA works with PyTorch on Windows + AMD GPU
3. If ZLUDA avoids the ROCm tensor transfer bug at specific failure points
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_zluda_installation():
    """Test if ZLUDA can be configured/installed."""
    logger.info("=" * 60)
    logger.info("Test 1: ZLUDA Installation/Configuration")
    logger.info("=" * 60)
    
    # ZLUDA is typically installed as a DLL that intercepts CUDA calls
    # Check if ZLUDA DLL is available
    zluda_dll_paths = [
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvcuda.dll"),
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "zluda.dll"),
        "zluda.dll",
    ]
    
    zluda_found = False
    for dll_path in zluda_dll_paths:
        if os.path.exists(dll_path):
            logger.info(f"✓ Found potential ZLUDA DLL: {dll_path}")
            zluda_found = True
        else:
            logger.debug(f"  Not found: {dll_path}")
    
    if not zluda_found:
        logger.warning("⚠ ZLUDA DLL not found in standard locations")
        logger.info("  ZLUDA typically needs to be installed separately")
        logger.info("  Check: https://github.com/vosen/ZLUDA")
        logger.info("  ZLUDA may require manual installation or environment setup")
    
    # Check environment variables that might indicate ZLUDA
    zluda_env_vars = ["ZLUDA_PATH", "CUDA_PATH", "CUDA_HOME"]
    for var in zluda_env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"  {var}={value}")
    
    return zluda_found

def test_rocm_detection():
    """Test current ROCm setup."""
    logger.info("=" * 60)
    logger.info("Test 2: Current ROCm Setup")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA/ROCm not available")
        return False
    
    logger.info(f"✓ CUDA/ROCm available: {torch.cuda.is_available()}")
    logger.info(f"  Device count: {torch.cuda.device_count()}")
    logger.info(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    if hasattr(torch.version, 'hip') and torch.version.hip:
        logger.info(f"  ROCm version: {torch.version.hip}")
        logger.info("  Current backend: ROCm (HIP)")
    else:
        logger.warning("  ROCm version not detected")
    
    return True

def test_zluda_tensor_transfers():
    """Test if ZLUDA (if available) avoids the ROCm tensor transfer bug."""
    logger.info("=" * 60)
    logger.info("Test 3: Tensor Transfer Test (ROCm Bug Check)")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA/ROCm not available for transfer test")
        return False
    
    device = torch.device("cuda")
    success_count = 0
    total_tests = 5
    
    # Test the same transfer methods that fail in training
    for i in range(total_tests):
        # Create a tensor with non-zero values (similar to training tensors)
        cpu_tensor = torch.randn(100, 100) + 1.0  # Ensure non-zero
        max_cpu = cpu_tensor.abs().max().item()
        
        if max_cpu < 1e-6:
            logger.error(f"✗ Test {i+1}: CPU tensor is already zeros!")
            continue
        
        # Test Method 1: Simple .to(device)
        try:
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
            max_gpu = gpu_tensor.abs().max().item()
            
            if max_gpu < 1e-6:
                logger.error(f"✗ Test {i+1}: ROCm bug detected! Tensor became zeros")
                logger.error(f"  CPU max: {max_cpu:.6e}, GPU max: {max_gpu:.6e}")
            else:
                logger.info(f"✓ Test {i+1}: Transfer successful (CPU max: {max_cpu:.6e}, GPU max: {max_gpu:.6e})")
                success_count += 1
        except Exception as e:
            logger.error(f"✗ Test {i+1}: Transfer failed with exception: {e}")
    
    if success_count == total_tests:
        logger.info(f"✓ All {total_tests} tensor transfers successful")
        logger.info("  Note: This doesn't guarantee ZLUDA is working - could be ROCm")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{total_tests} transfers successful")
        logger.error("  ROCm bug is present - ZLUDA might help if properly configured")
        return False

def test_zluda_specific_failure_points():
    """Test the specific failure points from move_to_gpu function."""
    logger.info("=" * 60)
    logger.info("Test 4: Specific Failure Point Tests")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda")
    
    # Test the 8 methods from move_to_gpu function
    methods = {
        "Method 1 (pinned memory)": lambda t: t.pin_memory().to(device, non_blocking=True),
        "Method 2 (direct .to())": lambda t: t.to(device, non_blocking=False),
        "Method 3 (copy_())": lambda t: torch.empty(t.shape, dtype=t.dtype, device=device).copy_(t, non_blocking=False),
        "Method 5 (numpy + tensor)": lambda t: torch.tensor(t.cpu().numpy(), dtype=t.dtype, device=device),
        "Method 8 (numpy + as_tensor)": lambda t: torch.as_tensor(t.detach().cpu().numpy(), device=device),
    }
    
    cpu_tensor = torch.randn(1000, 1000) + 1.0  # Large tensor like training
    max_cpu = cpu_tensor.abs().max().item()
    
    logger.info(f"Testing with tensor: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}, max={max_cpu:.6e}")
    
    results = {}
    for method_name, method_func in methods.items():
        try:
            gpu_tensor = method_func(cpu_tensor)
            torch.cuda.synchronize(device)
            max_gpu = gpu_tensor.abs().max().item()
            
            if max_gpu > 1e-6:
                logger.info(f"✓ {method_name}: SUCCESS (GPU max: {max_gpu:.6e})")
                results[method_name] = True
            else:
                logger.error(f"✗ {method_name}: FAILED - zeros (GPU max: {max_gpu:.6e})")
                results[method_name] = False
        except Exception as e:
            logger.error(f"✗ {method_name}: EXCEPTION - {e}")
            results[method_name] = False
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Results: {success_count}/{len(methods)} methods succeeded")
    
    return success_count > 0

def test_zluda_vs_rocm():
    """Test if we can detect whether ZLUDA or ROCm is being used."""
    logger.info("=" * 60)
    logger.info("Test 5: ZLUDA vs ROCm Detection")
    logger.info("=" * 60)
    
    # Check PyTorch backend
    if hasattr(torch.version, 'hip') and torch.version.hip:
        logger.info(f"  PyTorch backend: ROCm (HIP) version {torch.version.hip}")
        logger.info("  ZLUDA would need to intercept CUDA calls before PyTorch")
        logger.info("  This would require PyTorch built for CUDA, not ROCm")
        logger.warning("  ⚠ ZLUDA may not work with ROCm-built PyTorch")
        logger.warning("  ⚠ ZLUDA typically requires CUDA-built PyTorch")
        return False
    elif hasattr(torch.version, 'cuda') and torch.version.cuda:
        logger.info(f"  PyTorch backend: CUDA version {torch.version.cuda}")
        logger.info("  ✓ ZLUDA could potentially work with CUDA-built PyTorch")
        return True
    else:
        logger.warning("  Unknown PyTorch backend")
        return False

def main():
    """Run all ZLUDA compatibility tests."""
    logger.info("ZLUDA ROCm Compatibility Test")
    logger.info("=" * 60)
    logger.info("Testing ZLUDA as potential workaround for ROCm bug")
    logger.info("=" * 60)
    
    results = {
        "installation": test_zluda_installation(),
        "rocm_detection": test_rocm_detection(),
        "tensor_transfers": test_zluda_tensor_transfers(),
        "failure_points": test_zluda_specific_failure_points(),
        "zluda_vs_rocm": test_zluda_vs_rocm(),
    }
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/UNCERTAIN"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info("ZLUDA Assessment")
    logger.info("=" * 60)
    
    if results["zluda_vs_rocm"]:
        logger.info("⚠ ZLUDA requires CUDA-built PyTorch, but we have ROCm-built PyTorch")
        logger.info("  ZLUDA may not be compatible with current setup")
        logger.info("  Would need to switch to CUDA-built PyTorch (not recommended)")
    else:
        logger.info("⚠ Current PyTorch is ROCm-built")
        logger.info("  ZLUDA typically works with CUDA-built PyTorch")
        logger.info("  ZLUDA intercepts CUDA calls, but PyTorch is using ROCm/HIP directly")
        logger.info("  ZLUDA may not help in this configuration")
    
    logger.info("")
    logger.info("Recommendation:")
    logger.info("  ZLUDA is designed to translate CUDA→AMD, but PyTorch is already using ROCm")
    logger.info("  ZLUDA would need PyTorch built for CUDA, which defeats the purpose")
    logger.info("  Consider: Focus on ROCm-specific workarounds instead")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



