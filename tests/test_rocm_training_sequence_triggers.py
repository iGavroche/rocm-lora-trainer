"""Test to isolate which part of the training sequence triggers the ROCm bug.

The bug: Isolated tensor transfers work, but full training context fails.
This test progressively adds complexity to find the trigger.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_baseline_transfer():
    """Baseline: Simple CPU→GPU transfer (should work)."""
    logger.info("=" * 60)
    logger.info("Test 1: Baseline Transfer (Isolated)")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Baseline transfer works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return True
    else:
        logger.error(f"✗ Baseline transfer failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return False

def test_after_model_loaded():
    """Test: Transfer after loading a large model (simulating training setup)."""
    logger.info("=" * 60)
    logger.info("Test 2: Transfer After Model Loaded on GPU")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Simulate loading a large model (allocate GPU memory)
    logger.info("Allocating large tensor on GPU to simulate model...")
    model_sim = torch.randn(5000, 5000, device=device)  # ~100MB
    torch.cuda.synchronize(device)
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    logger.info(f"GPU memory: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
    
    # Now try transfer
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer after model load works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model_sim
        torch.cuda.empty_cache()
        return True
    else:
        logger.error(f"✗ Transfer after model load failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model_sim
        torch.cuda.empty_cache()
        return False

def test_after_forward_pass():
    """Test: Transfer after a forward pass (model operations)."""
    logger.info("=" * 60)
    logger.info("Test 3: Transfer After Forward Pass")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Create a simple model and do forward pass
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 100)
    ).to(device)
    
    # Forward pass
    input_tensor = torch.randn(10, 100, device=device)
    output = model(input_tensor)
    torch.cuda.synchronize(device)
    
    logger.info("Forward pass completed")
    
    # Now try transfer
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer after forward pass works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model, input_tensor, output
        torch.cuda.empty_cache()
        return True
    else:
        logger.error(f"✗ Transfer after forward pass failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model, input_tensor, output
        torch.cuda.empty_cache()
        return False

def test_after_backward_pass():
    """Test: Transfer after a backward pass (gradient computation)."""
    logger.info("=" * 60)
    logger.info("Test 4: Transfer After Backward Pass")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Create model and do backward pass
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 100)
    ).to(device)
    
    input_tensor = torch.randn(10, 100, device=device, requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize(device)
    
    logger.info("Backward pass completed")
    
    # Now try transfer
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer after backward pass works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model, input_tensor, output, loss
        torch.cuda.empty_cache()
        return True
    else:
        logger.error(f"✗ Transfer after backward pass failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        del model, input_tensor, output, loss
        torch.cuda.empty_cache()
        return False

def test_training_tensor_shapes():
    """Test: Transfer with exact tensor shapes from training."""
    logger.info("=" * 60)
    logger.info("Test 5: Transfer with Training Tensor Shapes")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Typical training tensor shapes (from WAN model)
    test_shapes = [
        (1, 16, 9, 64, 64),  # latents: B, C, F, H, W
        (1, 16, 9, 64, 64),  # noise
        (1, 16, 9, 64, 64),  # noisy_model_input
        (1,),  # timesteps
        (1, 20, 4096),  # t5 context
    ]
    
    all_passed = True
    for i, shape in enumerate(test_shapes):
        cpu_tensor = torch.randn(*shape) + 1.0
        max_cpu = cpu_tensor.abs().max().item()
        
        # Make contiguous (training does this)
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        max_gpu = gpu_tensor.abs().max().item()
        
        if max_gpu > 1e-6:
            logger.info(f"✓ Shape {shape}: works (CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e})")
        else:
            logger.error(f"✗ Shape {shape}: FAILED (CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e})")
            all_passed = False
    
    return all_passed

def test_with_accelerate_context():
    """Test: Transfer within Accelerate context."""
    logger.info("=" * 60)
    logger.info("Test 6: Transfer Within Accelerate Context")
    logger.info("=" * 60)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Prepare a dummy model with Accelerate
    model = torch.nn.Linear(100, 100)
    model = accelerator.prepare(model)
    
    logger.info(f"Accelerate device: {device}")
    
    # Try transfer
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer in Accelerate context works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return True
    else:
        logger.error(f"✗ Transfer in Accelerate context failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return False

def test_sequential_transfers():
    """Test: Multiple sequential transfers (simulating training loop)."""
    logger.info("=" * 60)
    logger.info("Test 7: Sequential Transfers (Training Loop Simulation)")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Simulate training loop: multiple transfers in sequence
    success_count = 0
    total = 10
    
    for i in range(total):
        cpu_tensor = torch.randn(100, 100) + 1.0
        max_cpu = cpu_tensor.abs().max().item()
        
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        max_gpu = gpu_tensor.abs().max().item()
        
        if max_gpu > 1e-6:
            success_count += 1
        else:
            logger.error(f"✗ Transfer {i+1} failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
            # This is the trigger - which transfer number fails?
            logger.error(f"  CRITICAL: Transfer {i+1} is where it fails!")
            break
        
        # Simulate some GPU operations between transfers
        if i < total - 1:
            dummy = torch.randn(10, 10, device=device)
            _ = dummy.sum()
            del dummy
    
    if success_count == total:
        logger.info(f"✓ All {total} sequential transfers worked")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{total} transfers worked")
        logger.error(f"  Failure occurred at transfer {success_count + 1}")
        return False

def test_memory_pressure():
    """Test: Transfer under GPU memory pressure."""
    logger.info("=" * 60)
    logger.info("Test 8: Transfer Under Memory Pressure")
    logger.info("=" * 60)
    
    device = torch.device("cuda")
    
    # Fill GPU memory to ~80%
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = int(total_memory * 0.8)
    
    logger.info(f"Filling GPU memory to ~80% ({target_memory / 1024**3:.2f} GB)")
    
    memory_blocks = []
    allocated = 0
    block_size = 100 * 1024 * 1024  # 100MB blocks
    
    while allocated < target_memory:
        try:
            block = torch.randn(block_size // 4, device=device)  # float32 = 4 bytes
            memory_blocks.append(block)
            allocated += block_size
        except RuntimeError as e:
            logger.info(f"Memory allocation stopped: {e}")
            break
    
    current_allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"GPU memory allocated: {current_allocated:.2f} GB")
    
    # Now try transfer
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    # Cleanup
    del memory_blocks
    torch.cuda.empty_cache()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer under memory pressure works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return True
    else:
        logger.error(f"✗ Transfer under memory pressure failed: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return False

def test_exact_training_sequence():
    """Test: Exact sequence from training (model load → prepare → transfer)."""
    logger.info("=" * 60)
    logger.info("Test 9: Exact Training Sequence")
    logger.info("=" * 60)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Step 1: Load model (simulate)
    logger.info("Step 1: Loading model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 100)
    )
    model = model.to(device)
    torch.cuda.synchronize(device)
    
    # Step 2: Prepare with Accelerate
    logger.info("Step 2: Preparing with Accelerate...")
    model = accelerator.prepare(model)
    
    # Step 3: Set training mode
    logger.info("Step 3: Setting training mode...")
    model.train()
    
    # Step 4: Do a forward/backward pass
    logger.info("Step 4: Forward/backward pass...")
    input_tensor = torch.randn(10, 100, device=device, requires_grad=True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize(device)
    
    # Step 5: Now try transfer (this is where it fails in training)
    logger.info("Step 5: Attempting tensor transfer...")
    cpu_tensor = torch.randn(100, 100) + 1.0
    max_cpu = cpu_tensor.abs().max().item()
    
    gpu_tensor = cpu_tensor.to(device, non_blocking=False)
    max_gpu = gpu_tensor.abs().max().item()
    
    if max_gpu > 1e-6:
        logger.info(f"✓ Transfer in exact training sequence works: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        return True
    else:
        logger.error(f"✗ Transfer in exact training sequence FAILED: CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
        logger.error("  This sequence reproduces the bug!")
        return False

def main():
    """Run all tests to isolate the trigger."""
    logger.info("ROCm Training Sequence Trigger Isolation Test")
    logger.info("=" * 60)
    logger.info("Finding what triggers the ROCm tensor transfer bug")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA/ROCm not available")
        return 1
    
    results = {
        "baseline": test_baseline_transfer(),
        "after_model_loaded": test_after_model_loaded(),
        "after_forward": test_after_forward_pass(),
        "after_backward": test_after_backward_pass(),
        "training_shapes": test_training_tensor_shapes(),
        "accelerate_context": test_with_accelerate_context(),
        "sequential": test_sequential_transfers(),
        "memory_pressure": test_memory_pressure(),
        "exact_sequence": test_exact_training_sequence(),
    }
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    # Analysis
    logger.info("=" * 60)
    logger.info("Analysis")
    logger.info("=" * 60)
    
    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        logger.error(f"Failed tests: {failed_tests}")
        logger.error("These are the potential triggers for the ROCm bug")
    else:
        logger.info("All isolated tests passed - bug only occurs in full training context")
        logger.info("This suggests the trigger is something more complex:")
        logger.info("  - Specific combination of operations")
        logger.info("  - DataLoader worker context")
        logger.info("  - Specific tensor properties (non-contiguous, views, etc.)")
        logger.info("  - Timing/race condition")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())



