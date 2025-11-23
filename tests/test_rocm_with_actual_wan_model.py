"""Test ROCm bug with actual WAN model and cache files.

This test uses the real WAN model and cache loading to see if that triggers the bug.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_with_actual_wan_model():
    """Test transfers with actual WAN model loaded."""
    logger.info("=" * 60)
    logger.info("Test: Transfer with Actual WAN Model")
    logger.info("=" * 60)
    
    try:
        from src.musubi_tuner.wan.modules.model import load_wan_model
        from src.musubi_tuner.wan.configs import WAN_CONFIGS
    except ImportError as e:
        logger.error(f"Cannot import WAN model: {e}")
        return False
    
    accelerator = Accelerator()
    device = accelerator.device
    config = WAN_CONFIGS['i2v-A14B']
    
    logger.info("Loading WAN model...")
    try:
        model = load_wan_model(
            config,
            device,
            r'.\models\wan\wan2.2_i2v_low_noise_14B_fp16.safetensors',
            'xformers',
            False,
            'cpu',
            torch.float16,
            False,
            None,
            None,
            False,
            False
        )
        logger.info("WAN model loaded")
        
        # Move model to GPU
        model = model.to(device)
        torch.cuda.synchronize(device)
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"GPU memory after model load: {allocated:.2f} GB")
        
        # Now try transfer
        cpu_tensor = torch.randn(1, 16, 9, 64, 64) + 1.0  # Training tensor shape
        max_cpu = cpu_tensor.abs().max().item()
        
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        max_gpu = gpu_tensor.abs().max().item()
        
        if max_gpu > 1e-6:
            logger.info(f"✓ Transfer with WAN model works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
            return True
        else:
            logger.error(f"✗ Transfer with WAN model FAILED: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
            logger.error("  This is the trigger!")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_cache_loading():
    """Test transfers after loading from cache files."""
    logger.info("=" * 60)
    logger.info("Test: Transfer After Cache File Loading")
    logger.info("=" * 60)
    
    try:
        from musubi_tuner.dataset.image_video_dataset import ImageVideoDataset
        from musubi_tuner.dataset.config_utils import read_config_from_file
        import argparse
    except ImportError as e:
        logger.error(f"Cannot import dataset modules: {e}")
        return False
    
    # Check if cache files exist
    cache_dir = r".\cache"
    if not os.path.exists(cache_dir):
        logger.warning("Cache directory not found - skipping cache test")
        return None
    
    logger.info("Cache directory found - this test would load actual cache files")
    logger.info("Skipping full cache test (requires dataset setup)")
    return None

def test_with_dataloader_context():
    """Test transfers within DataLoader context."""
    logger.info("=" * 60)
    logger.info("Test: Transfer Within DataLoader Context")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return tensor on CPU (like cache files)
            return {
                "latents": torch.randn(1, 16, 9, 64, 64) + 1.0,
            }
    
    dataset = SimpleDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # Same as training
        pin_memory=False,
    )
    
    # Iterate through DataLoader and test transfers
    success_count = 0
    total = len(dataset)
    
    for i, batch in enumerate(dataloader):
        cpu_tensor = batch["latents"]
        max_cpu = cpu_tensor.abs().max().item()
        
        # Transfer (this is where it fails in training)
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        max_gpu = gpu_tensor.abs().max().item()
        
        if max_gpu > 1e-6:
            success_count += 1
        else:
            logger.error(f"✗ Transfer {i+1} in DataLoader FAILED: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
            logger.error(f"  This is the trigger - DataLoader batch {i+1}!")
            return False
    
    if success_count == total:
        logger.info(f"✓ All {total} DataLoader transfers worked")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{total} DataLoader transfers worked")
        return False

def main():
    """Run tests with actual components."""
    logger.info("ROCm Bug Trigger Test - Actual Components")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA/ROCm not available")
        return 1
    
    results = {}
    
    # Test with DataLoader (most likely trigger)
    results["dataloader"] = test_with_dataloader_context()
    
    # Test with actual WAN model
    results["wan_model"] = test_with_actual_wan_model()
    
    # Test with cache (if available)
    cache_result = test_with_cache_loading()
    if cache_result is not None:
        results["cache"] = cache_result
    
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
        logger.error("These components trigger the ROCm bug")
    else:
        logger.info("=" * 60)
        logger.info("All component tests passed")
        logger.info("Bug trigger is still unknown - may be:")
        logger.info("  - Specific combination of all components together")
        logger.info("  - Actual cache file format/structure")
        logger.info("  - Timing/race condition in full training loop")
    
    return 0 if not failed_tests else 1

if __name__ == "__main__":
    sys.exit(main())
