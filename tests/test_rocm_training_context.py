#!/usr/bin/env python3
"""
Test the EXACT training context to find what's different.

This test replicates the exact training setup including Accelerate, model loading, etc.
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_training_context():
    """Test the exact training context"""
    print("=" * 80)
    print("ROCm Training Context Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    print(f"Device: cuda:0")
    print()
    
    try:
        # Step 1: Setup Accelerate (same as training)
        from accelerate import Accelerator
        accelerator = Accelerator()
        print(f"Step 1: Accelerator setup")
        print(f"  Device: {accelerator.device}")
        
        # Step 2: Load dataset (same as training)
        from musubi_tuner.dataset.config_utils import (
            load_user_config, BlueprintGenerator, ConfigSanitizer, 
            generate_dataset_group_by_blueprint
        )
        import argparse
        
        config_path = Path("dataset.toml")
        if not config_path.exists():
            print("ERROR: dataset.toml not found")
            return False
        
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
        
        # Step 3: Create DataLoader WITHOUT preparing it with Accelerate (same as training)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collator
        )
        print(f"Step 2: Created DataLoader (NOT prepared with Accelerate)")
        
        # Step 4: Get batch
        batch = next(iter(train_dataloader))
        print(f"Step 3: Got batch, keys: {list(batch.keys())}")
        
        # Step 5: Check latents on CPU
        latents_cpu = batch["latents"]
        cpu_max = latents_cpu.abs().max().item()
        print(f"Step 4: Latents on CPU")
        print(f"  Shape: {latents_cpu.shape}, dtype: {latents_cpu.dtype}, device: {latents_cpu.device}")
        print(f"  Max: {cpu_max:.6e}")
        
        if cpu_max < 1e-6:
            print("  ❌ FAILED: Latents are zeros on CPU!")
            return False
        
        # Step 6: Move to GPU using accelerator.device (same as training)
        print(f"\nStep 5: Moving latents to GPU using accelerator.device ({accelerator.device})...")
        
        # Test direct .to() first (this should work based on other tests)
        latents_gpu = latents_cpu.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max = latents_gpu.abs().max().item()
        
        print(f"  Direct .to(accelerator.device):")
        print(f"    GPU max: {gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Direct .to() produces zeros!")
            print("  This is the exact failure in training.")
            return False
        else:
            print("  ✅ PASSED: Direct .to() works")
        
        # Step 7: Test with move_to_gpu function (exact copy from training)
        print(f"\nStep 6: Testing with move_to_gpu function...")
        
        def move_to_gpu(value, name="", step=0):
            """Exact copy from training"""
            if not isinstance(value, torch.Tensor):
                return value
            
            if value.device.type == "cpu":
                max_val_cpu = value.abs().max().item()
                if max_val_cpu < 1e-6:
                    return value.to(accelerator.device, non_blocking=False)
                
                # Method 2: Try direct .to() with non_blocking=False
                try:
                    tensor_gpu = value.to(accelerator.device, non_blocking=False)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    
                    if max_val_gpu > 1e-6:
                        return tensor_gpu
                except Exception as e:
                    print(f"    Method 2 exception: {e}")
                
                # If Method 2 fails, try others...
                # (simplified for testing)
                return None
            else:
                return value
        
        latents_gpu_func = move_to_gpu(latents_cpu, "latents", step=0)
        if latents_gpu_func is None:
            print("  ❌ FAILED: move_to_gpu function failed!")
            return False
        
        gpu_max_func = latents_gpu_func.abs().max().item()
        print(f"  move_to_gpu result: GPU max={gpu_max_func:.6e}")
        
        if gpu_max_func < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: move_to_gpu produces zeros!")
            return False
        else:
            print("  ✅ PASSED: move_to_gpu works")
        
        # Step 8: Test after model loading (simulate model being on GPU)
        print(f"\nStep 7: Testing after simulating model memory usage...")
        
        # Allocate some memory to simulate model
        dummy_model_tensors = []
        for i in range(5):
            dummy = torch.randn((1000, 1000), device=accelerator.device)
            dummy_model_tensors.append(dummy)
        
        # Now try moving latents again
        latents_gpu_after_model = latents_cpu.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_after = latents_gpu_after_model.abs().max().item()
        
        print(f"  After model memory allocation:")
        print(f"    GPU max: {gpu_max_after:.6e}")
        
        if gpu_max_after < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Transfer fails after model memory allocation!")
            return False
        else:
            print("  ✅ PASSED: Transfer works even after model memory")
        
        # Clean up
        del dummy_model_tensors
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nThe training context works correctly.")
        print("If training still fails, check:")
        print("  1. The exact sequence in the training loop")
        print("  2. Model forward pass operations")
        print("  3. Mixed precision context")
        print("  4. Gradient checkpointing")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_context()
    sys.exit(0 if success else 1)


