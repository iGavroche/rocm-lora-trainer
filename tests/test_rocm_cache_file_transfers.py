"""Test ROCm bug with actual cache files.

This test loads actual cache files and tests transfers to see if that triggers the bug.
"""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_cache_file_transfers():
    """Test transfers with tensors loaded from actual cache files."""
    logger.info("=" * 60)
    logger.info("Test: Transfer with Cache File Tensors")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Find cache files
    cache_dir = r".\cache"
    if not os.path.exists(cache_dir):
        logger.warning("Cache directory not found - cannot test with actual cache files")
        return None
    
    # Look for WAN cache files
    cache_pattern = os.path.join(cache_dir, "*_WAN.safetensors")
    cache_files = glob.glob(cache_pattern)
    
    if not cache_files:
        logger.warning("No WAN cache files found - cannot test")
        return None
    
    logger.info(f"Found {len(cache_files)} cache files")
    logger.info(f"Testing with: {cache_files[0]}")
    
    # Load tensor from cache file (to CPU, like training does)
    cache_file = cache_files[0]
    try:
        with safe_open(cache_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            logger.info(f"Cache file keys: {list(keys)[:5]}...")  # Show first 5
            
            # Load first tensor
            first_key = list(keys)[0]
            logger.info(f"Loading tensor '{first_key}' from cache...")
            
            # Load to CPU (like training does)
            cpu_tensor = f.get_tensor(first_key, device=torch.device("cpu"), dtype=torch.float32)
            max_cpu = cpu_tensor.abs().max().item()
            
            logger.info(f"CPU tensor: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}, max={max_cpu:.6e}")
            
            if max_cpu < 1e-6:
                logger.error(f"✗ Cache tensor is already zeros! Cache file may be corrupted")
                return False
            
            # Check tensor properties
            logger.info(f"Tensor properties: contiguous={cpu_tensor.is_contiguous()}, pinned={cpu_tensor.is_pinned()}")
            
            # Make contiguous (training does this)
            if not cpu_tensor.is_contiguous():
                logger.info("Making tensor contiguous...")
                cpu_tensor = cpu_tensor.contiguous()
            
            # Now try transfer to GPU (this is where it fails in training)
            logger.info("Transferring to GPU...")
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
            max_gpu = gpu_tensor.abs().max().item()
            
            logger.info(f"GPU tensor: shape={gpu_tensor.shape}, dtype={gpu_tensor.dtype}, max={max_gpu:.6e}")
            
            if max_gpu > 1e-6:
                logger.info(f"✓ Transfer from cache file works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
                return True
            else:
                logger.error(f"✗ Transfer from cache file FAILED: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
                logger.error("  THIS IS THE TRIGGER - cache file tensors!")
                return False
                
    except Exception as e:
        logger.error(f"Failed to load cache file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_multiple_cache_tensors():
    """Test transfers with multiple tensors from cache files."""
    logger.info("=" * 60)
    logger.info("Test: Multiple Cache Tensor Transfers")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    cache_dir = r".\cache"
    cache_pattern = os.path.join(cache_dir, "*_WAN.safetensors")
    cache_files = glob.glob(cache_pattern)
    
    if not cache_files:
        logger.warning("No cache files found")
        return None
    
    success_count = 0
    total = 0
    
    # Test first 5 cache files
    for cache_file in cache_files[:5]:
        try:
            with safe_open(cache_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                
                # Test first tensor from each file
                if keys:
                    first_key = keys[0]
                    cpu_tensor = f.get_tensor(first_key, device=torch.device("cpu"), dtype=torch.float32)
                    
                    if not cpu_tensor.is_contiguous():
                        cpu_tensor = cpu_tensor.contiguous()
                    
                    max_cpu = cpu_tensor.abs().max().item()
                    
                    if max_cpu > 1e-6:
                        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
                        max_gpu = gpu_tensor.abs().max().item()
                        
                        total += 1
                        if max_gpu > 1e-6:
                            success_count += 1
                        else:
                            logger.error(f"✗ Failed: {os.path.basename(cache_file)}/{first_key}")
                            logger.error(f"  CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        except Exception as e:
            logger.warning(f"Error with {cache_file}: {e}")
    
    if total == 0:
        logger.warning("No tensors tested")
        return None
    
    logger.info(f"Results: {success_count}/{total} transfers successful")
    
    if success_count == total:
        logger.info("✓ All cache tensor transfers worked")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{total} cache tensor transfers worked")
        return False

def test_cache_tensor_properties():
    """Analyze properties of cache tensors that might trigger the bug."""
    logger.info("=" * 60)
    logger.info("Test: Cache Tensor Property Analysis")
    logger.info("=" * 60)
    
    cache_dir = r".\cache"
    cache_pattern = os.path.join(cache_dir, "*_WAN.safetensors")
    cache_files = glob.glob(cache_pattern)
    
    if not cache_files:
        logger.warning("No cache files found")
        return None
    
    cache_file = cache_files[0]
    
    try:
        with safe_open(cache_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            logger.info(f"Analyzing {len(keys)} tensors from cache file...")
            
            properties = {
                "shapes": set(),
                "dtypes": set(),
                "sizes": [],
                "contiguous": [],
                "pinned": [],
            }
            
            for key in keys[:10]:  # Analyze first 10
                tensor = f.get_tensor(key, device=torch.device("cpu"), dtype=torch.float32)
                
                properties["shapes"].add(tensor.shape)
                properties["dtypes"].add(tensor.dtype)
                properties["sizes"].append(tensor.numel())
                properties["contiguous"].append(tensor.is_contiguous())
                properties["pinned"].append(tensor.is_pinned())
            
            logger.info(f"Shapes: {list(properties['shapes'])[:5]}")
            logger.info(f"Dtypes: {list(properties['dtypes'])}")
            logger.info(f"Size range: {min(properties['sizes'])} - {max(properties['sizes'])}")
            logger.info(f"Contiguous: {sum(properties['contiguous'])}/{len(properties['contiguous'])}")
            logger.info(f"Pinned: {sum(properties['pinned'])}/{len(properties['pinned'])}")
            
            # Check for unusual properties
            if not all(properties["contiguous"]):
                logger.warning("⚠ Some cache tensors are non-contiguous - this might trigger the bug")
            
            return True
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def main():
    """Run cache file transfer tests."""
    logger.info("ROCm Cache File Transfer Test")
    logger.info("=" * 60)
    logger.info("Testing if cache file tensors trigger the ROCm bug")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA/ROCm not available")
        return 1
    
    results = {}
    
    # Test single cache tensor
    results["single_cache"] = test_cache_file_transfers()
    
    # Test multiple cache tensors
    results["multiple_cache"] = test_multiple_cache_tensors()
    
    # Analyze cache tensor properties
    results["properties"] = test_cache_tensor_properties()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        if passed is None:
            status = "⚠ SKIP"
        else:
            status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    failed_tests = [name for name, passed in results.items() if passed is False]
    if failed_tests:
        logger.error("=" * 60)
        logger.error("TRIGGER FOUND!")
        logger.error(f"Failed tests: {failed_tests}")
        logger.error("Cache file tensors trigger the ROCm bug!")
    else:
        logger.info("=" * 60)
        logger.info("Cache file transfers work in isolation")
        logger.info("Bug trigger is still unknown - may require full training context")
    
    return 0 if not failed_tests else 1

if __name__ == "__main__":
    sys.exit(main())



