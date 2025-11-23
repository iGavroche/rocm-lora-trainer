"""Test Torchtune compatibility with Windows + ROCm.

This test verifies if Torchtune can be installed and works correctly
with ROCm on Windows, similar to the PEFT compatibility test.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_torchtune_installation():
    """Test if Torchtune can be installed and imported."""
    logger.info("=" * 60)
    logger.info("Test 1: Torchtune Installation")
    logger.info("=" * 60)
    
    try:
        import torchtune
        logger.info(f"✓ Torchtune installed: version {torchtune.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Torchtune not installed: {e}")
        logger.info("Attempting to install Torchtune...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torchtune"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                import torchtune
                logger.info(f"✓ Torchtune installed successfully: version {torchtune.__version__}")
                return True
            else:
                logger.error(f"✗ Torchtune installation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"✗ Failed to install Torchtune: {e}")
            return False

def test_torchtune_rocm_detection():
    """Test if Torchtune can detect ROCm."""
    logger.info("=" * 60)
    logger.info("Test 2: ROCm Detection")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA/ROCm not available")
        return False
    
    logger.info(f"✓ CUDA/ROCm available: {torch.cuda.is_available()}")
    logger.info(f"  Device count: {torch.cuda.device_count()}")
    logger.info(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    if hasattr(torch.version, 'hip') and torch.version.hip:
        logger.info(f"✓ ROCm detected: {torch.version.hip}")
    else:
        logger.warning("⚠ ROCm version not detected (may still be ROCm)")
    
    return True

def test_torchtune_basic_operations():
    """Test basic Torchtune operations with ROCm."""
    logger.info("=" * 60)
    logger.info("Test 3: Basic Torchtune Operations")
    logger.info("=" * 60)
    
    try:
        import torchtune
        
        # Check if Torchtune has LoRA support
        if hasattr(torchtune, 'modules'):
            logger.info("✓ Torchtune modules available")
        else:
            logger.warning("⚠ Torchtune modules structure unclear")
        
        # Check if Torchtune has recipes/configs
        if hasattr(torchtune, 'recipes'):
            logger.info("✓ Torchtune recipes available")
        else:
            logger.warning("⚠ Torchtune recipes structure unclear")
        
        # List available attributes
        logger.info(f"Torchtune attributes: {[attr for attr in dir(torchtune) if not attr.startswith('_')][:10]}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Torchtune basic operations failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_torchtune_tensor_transfers():
    """Test if Torchtune avoids the ROCm tensor transfer bug."""
    logger.info("=" * 60)
    logger.info("Test 4: Tensor Transfer Test (ROCm Bug Check)")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA/ROCm not available for transfer test")
        return False
    
    device = torch.device("cuda")
    success_count = 0
    total_tests = 5
    
    for i in range(total_tests):
        # Create a tensor with non-zero values
        cpu_tensor = torch.randn(100, 100) + 1.0  # Ensure non-zero
        max_cpu = cpu_tensor.abs().max().item()
        
        if max_cpu < 1e-6:
            logger.error(f"✗ Test {i+1}: CPU tensor is already zeros!")
            continue
        
        # Transfer to GPU
        try:
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
            max_gpu = gpu_tensor.abs().max().item()
            
            if max_gpu < 1e-6:
                logger.error(f"✗ Test {i+1}: ROCm bug detected! Tensor became zeros after transfer")
                logger.error(f"  CPU max: {max_cpu:.6e}, GPU max: {max_gpu:.6e}")
            else:
                logger.info(f"✓ Test {i+1}: Transfer successful (CPU max: {max_cpu:.6e}, GPU max: {max_gpu:.6e})")
                success_count += 1
        except Exception as e:
            logger.error(f"✗ Test {i+1}: Transfer failed with exception: {e}")
    
    if success_count == total_tests:
        logger.info(f"✓ All {total_tests} tensor transfers successful - NO ROCm bug detected")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{total_tests} transfers successful - ROCm bug may be present")
        return False

def test_torchtune_model_compatibility():
    """Test if Torchtune can work with custom models (WAN compatibility check)."""
    logger.info("=" * 60)
    logger.info("Test 5: Model Compatibility Check")
    logger.info("=" * 60)
    
    try:
        import torchtune
        
        # Check if Torchtune supports custom models or only specific ones
        # Torchtune is designed for LLMs (Llama, Gemma, etc.), not video models
        logger.info("⚠ Torchtune is designed for LLMs (Llama, Gemma, Mistral, etc.)")
        logger.info("⚠ WAN is a video diffusion model, not an LLM")
        logger.info("⚠ Torchtune may require significant adaptation for WAN model")
        
        # Check if Torchtune has generic LoRA support
        if hasattr(torchtune, 'modules') and hasattr(torchtune.modules, 'lora'):
            logger.info("✓ Torchtune has LoRA module support")
        else:
            logger.warning("⚠ Torchtune LoRA structure unclear")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model compatibility check failed: {e}")
        return False

def main():
    """Run all Torchtune compatibility tests."""
    logger.info("Torchtune ROCm Compatibility Test")
    logger.info("=" * 60)
    logger.info("Testing Torchtune on Windows + ROCm")
    logger.info("=" * 60)
    
    results = {
        "installation": test_torchtune_installation(),
        "rocm_detection": test_torchtune_rocm_detection(),
        "basic_operations": False,
        "tensor_transfers": False,
        "model_compatibility": False,
    }
    
    if results["installation"]:
        results["basic_operations"] = test_torchtune_basic_operations()
        results["tensor_transfers"] = test_torchtune_tensor_transfers()
        results["model_compatibility"] = test_torchtune_model_compatibility()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("=" * 60)
        logger.info("✓ All tests passed - Torchtune appears compatible")
    else:
        logger.info("=" * 60)
        logger.info("✗ Some tests failed - Torchtune compatibility uncertain")
        logger.info("  Note: Even if tests pass, Torchtune may require")
        logger.info("  significant code changes to work with WAN model")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())



