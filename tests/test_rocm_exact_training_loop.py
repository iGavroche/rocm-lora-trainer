#!/usr/bin/env python3
"""
Test that replicates the EXACT training loop sequence to find the issue.

This test:
1. Replicates the exact training loop code
2. Includes all the same checks and operations
3. Tests if the issue reproduces

Run with: python tests/test_rocm_exact_training_loop.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_exact_training_loop():
    """Test the exact training loop sequence"""
    print("=" * 80)
    print("ROCm Exact Training Loop Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Setup exactly like training
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
        
        # Replicate the exact move_to_gpu function from training
        def move_to_gpu(value, name="", step=0):
            if not isinstance(value, torch.Tensor):
                return value
            
            if value.device.type == "cpu":
                max_val_cpu = value.abs().max().item()
                if max_val_cpu < 1e-6:
                    return value.to(accelerator.device, non_blocking=False)
                
                if not value.is_contiguous():
                    value = value.contiguous()
                
                value_clone = value.clone()
                max_val_clone = value_clone.abs().max().item()
                
                if max_val_clone < 1e-6:
                    raise RuntimeError(f"Tensor {name} became zeros after clone()")
                
                # Try Method 1: Pinned memory
                try:
                    if not value_clone.is_pinned():
                        pinned_value = value_clone.pin_memory()
                    else:
                        pinned_value = value_clone
                    tensor_gpu = pinned_value.to(accelerator.device, non_blocking=True)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    if max_val_gpu > 1e-6:
                        return tensor_gpu
                except:
                    pass
                
                # Try Method 2: Direct .to()
                try:
                    tensor_gpu = value_clone.to(accelerator.device, non_blocking=False)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    if max_val_gpu > 1e-6:
                        return tensor_gpu
                except:
                    pass
                
                # All methods failed
                raise RuntimeError(f"All transfer methods failed for {name}")
            else:
                return value
        
        # Replicate the exact training loop
        print("Replicating exact training loop sequence...")
        print("-" * 80)
        
        for step, batch in enumerate(train_dataloader):
            if step >= 3:  # Test first 3 steps
                break
            
            print(f"\nStep {step}:")
            
            # Exact same checks as training
            if step < 3:
                print(f"  Batch keys: {list(batch.keys())}")
                if "latents" not in batch:
                    print(f"  ❌ ERROR: 'latents' key not found!")
                    continue
            
            # Exact same transfer code as training
            if batch["latents"].device.type == "cuda":
                print(f"  Latents already on GPU, skipping transfer")
                latents = batch["latents"]
            else:
                print(f"  Transferring latents to GPU...")
                try:
                    latents = move_to_gpu(batch["latents"], "latents", step)
                    gpu_max = latents.abs().max().item()
                    cpu_max = batch["latents"].abs().max().item()
                    print(f"    ✅ SUCCESS: CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e}")
                except Exception as e:
                    print(f"    ❌ FAILED: {e}")
                    return False
            
            # Check if latents are zeros
            max_val = latents.abs().max().item()
            if max_val < 1e-6:
                print(f"  ❌ CRITICAL: Latents are zeros after transfer!")
                return False
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL STEPS PASSED")
        print("\nThe exact training loop sequence works correctly.")
        print("If training still fails, check:")
        print("  1. The actual model forward pass")
        print("  2. Something that happens after the transfer")
        print("  3. A race condition or timing issue")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_training_loop()
    sys.exit(0 if success else 1)


