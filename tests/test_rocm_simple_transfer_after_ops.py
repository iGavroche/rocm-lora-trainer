#!/usr/bin/env python3
"""
Simple test: Does a basic GPU operation corrupt subsequent transfers?

This is a minimal test to check if any GPU operation causes issues.
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value
import signal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def test_simple_transfer_after_ops():
    """Minimal test with timeout"""
    print("=" * 80)
    print("ROCm Simple Transfer After Operations Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        from musubi_tuner.dataset.config_utils import (
            load_user_config, BlueprintGenerator, ConfigSanitizer, 
            generate_dataset_group_by_blueprint
        )
        import argparse
        
        config_path = Path("dataset.toml")
        args = argparse.Namespace()
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        user_config = load_user_config(str(config_path))
        blueprint = blueprint_generator.generate(user_config, args, architecture="wan")
        
        shared_epoch = Value('i', 0)
        train_dataset_group = generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=True,
            num_timestep_buckets=None,
            shared_epoch=shared_epoch
        )
        
        dataset = train_dataset_group.datasets[0]
        dataset.current_epoch = 0
        
        from musubi_tuner.hv_train_network import collator_class
        current_epoch = Value('i', 0)
        collator = collator_class(current_epoch, dataset)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collator
        )
        
        # Get batch
        batch = next(iter(train_dataloader))
        latents = batch["latents"]
        cpu_max = latents.abs().max().item()
        print(f"Initial CPU max: {cpu_max:.6e}")
        
        # Transfer to GPU
        if not latents.is_contiguous():
            latents = latents.contiguous()
        latents_gpu = latents.clone().pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_1 = latents_gpu.abs().max().item()
        print(f"After transfer GPU max: {gpu_max_1:.6e}")
        
        if gpu_max_1 < 1e-6 and cpu_max > 1e-6:
            print("[FAILED] Initial transfer failed")
            return False
        
        # Simple GPU operation
        print("\nPerforming simple GPU operation (multiply by 2)...")
        result = latents_gpu * 2.0
        result_max = result.abs().max().item()
        print(f"Result max: {result_max:.6e}")
        
        # Try another transfer after operation
        print("\nTesting transfer after GPU operation...")
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max_2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_gpu = latents2.clone().pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_2 = latents2_gpu.abs().max().item()
        
        print(f"CPU max: {cpu_max_2:.6e}, GPU max: {gpu_max_2:.6e}")
        if gpu_max_2 < 1e-6 and cpu_max_2 > 1e-6:
            print("[FAILED] Transfer failed after GPU operation")
            return False
        
        print("\n[PASS] All transfers succeeded")
        return True
        
    except TimeoutError:
        print("\n[ERROR] Test timed out")
        return False
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_transfer_after_ops()
    sys.exit(0 if success else 1)

