"""Test exact training sequence to reproduce the ROCm bug.

This test replicates the exact sequence from the training loop to find the trigger.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_exact_training_sequence():
    """Replicate the exact training sequence from hv_train_network.py."""
    logger.info("=" * 60)
    logger.info("Test: Exact Training Sequence Replication")
    logger.info("=" * 60)
    
    try:
        from src.musubi_tuner.wan.modules.model import load_wan_model
        from src.musubi_tuner.wan.configs import WAN_CONFIGS
    except ImportError as e:
        logger.error(f"Cannot import required modules: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    accelerator = Accelerator()
    device = accelerator.device
    config = WAN_CONFIGS['i2v-A14B']
    
    # Step 1: Load model (exact sequence from training)
    logger.info("Step 1: Loading WAN model...")
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
    logger.info("Model loaded")
    
    # Step 2: Move model to GPU and set training mode
    logger.info("Step 2: Moving model to GPU and setting training mode...")
    model = model.to(device)
    model.train()
    torch.cuda.synchronize(device)
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"GPU memory after model load: {allocated:.2f} GB")
    
    # Step 3: Prepare model with Accelerate (simulating training setup)
    logger.info("Step 3: Preparing model with Accelerate...")
    model = accelerator.prepare(model)
    logger.info("Model prepared")
    
    # Step 4: Create a simple dataset (simulating cache-loaded tensors)
    logger.info("Step 4: Creating dataset (simulating cache files)...")
    
    class TrainingDataset(torch.utils.data.Dataset):
        def __init__(self, size=5):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return tensors on CPU (like cache files)
            return {
                "latents": torch.randn(1, 16, 9, 64, 64) + 1.0,  # Ensure non-zero
                "latents_image": torch.randn(1, 20, 9, 64, 64) + 1.0,
                "t5": [torch.randn(1, 20, 4096) + 1.0],
                "timesteps": torch.tensor([500.0]),
            }
    
    dataset = TrainingDataset(size=5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    
    # Step 5: Training loop (exact sequence)
    logger.info("Step 5: Starting training loop simulation...")
    
    success_count = 0
    failure_step = None
    
    for step, batch in enumerate(dataloader):
        logger.info(f"  Processing step {step}...")
        
        # This is the exact sequence from training
        # Check latents on CPU
        if "latents" not in batch:
            logger.error(f"Step {step}: 'latents' key not found!")
            continue
        
        cpu_latents = batch["latents"]
        max_cpu = cpu_latents.abs().max().item()
        
        if max_cpu < 1e-6:
            logger.error(f"Step {step}: CPU latents are already zeros!")
            continue
        
        logger.info(f"  Step {step}: CPU latents max={max_cpu:.6e}")
        
        # Make contiguous (training does this)
        if not cpu_latents.is_contiguous():
            cpu_latents = cpu_latents.contiguous()
        
        # Transfer to GPU (this is where it fails in training)
        logger.info(f"  Step {step}: Transferring to GPU...")
        gpu_latents = cpu_latents.to(device, non_blocking=False)
        max_gpu = gpu_latents.abs().max().item()
        
        logger.info(f"  Step {step}: GPU latents max={max_gpu:.6e}")
        
        if max_gpu > 1e-6:
            success_count += 1
            logger.info(f"  ✓ Step {step}: Transfer successful")
        else:
            logger.error(f"  ✗ Step {step}: Transfer FAILED!")
            logger.error(f"    CPU max={max_cpu:.6e}, GPU max={max_gpu:.6e}")
            failure_step = step
            break
        
        # Simulate forward pass (might trigger something)
        if step < len(dataloader) - 1:  # Don't do this on last step
            try:
                with torch.no_grad():
                    # Simple forward pass simulation (WAN model forward signature)
                    dummy_input = [gpu_latents]  # List format
                    dummy_t = torch.tensor([500.0], device=device)
                    # WAN model forward might need more args, so catch exceptions
                    try:
                        _ = model(dummy_input, t=dummy_t)
                    except TypeError:
                        # Model forward needs different args - skip
                        pass
                torch.cuda.synchronize(device)
            except Exception as e:
                logger.debug(f"  Forward pass simulation skipped: {e}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Training Sequence Test Results")
    logger.info("=" * 60)
    logger.info(f"Successful transfers: {success_count}/{len(dataset)}")
    
    if failure_step is not None:
        logger.error(f"✗ FAILURE at step {failure_step}")
        logger.error("  This reproduces the ROCm bug!")
        logger.error("  The trigger is in the training sequence at this step")
        return False
    elif success_count == len(dataset):
        logger.info("✓ All transfers successful")
        logger.info("  Bug not reproduced - trigger is something else")
        return True
    else:
        logger.error(f"✗ Only {success_count}/{len(dataset)} transfers worked")
        return False

def main():
    """Run exact training sequence test."""
    logger.info("ROCm Exact Training Sequence Test")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA/ROCm not available")
        return 1
    
    result = test_exact_training_sequence()
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
