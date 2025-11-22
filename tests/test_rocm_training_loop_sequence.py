#!/usr/bin/env python3
"""
Test the EXACT sequence of operations in the training loop to find where tensors become zeros.

This test simulates the training loop step-by-step to identify the exact point of failure.
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_training_loop_sequence():
    """Test the exact sequence of operations in training loop"""
    print("=" * 80)
    print("ROCm Training Loop Sequence Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Step 1: Load dataset (same as training)
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
        
        # Step 2: Create DataLoader (same as training)
        from musubi_tuner.hv_train_network import collator_class
        current_epoch = Value('i', 0)
        collator = collator_class(current_epoch, dataset)
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,  # Don't shuffle for reproducibility
            num_workers=0,
            pin_memory=False,
            collate_fn=collator
        )
        
        print("Step 1: Created DataLoader")
        
        # Step 3: Get first batch (same as training)
        batch = next(iter(train_dataloader))
        print(f"Step 2: Got batch, keys: {list(batch.keys())}")
        
        # Check latents on CPU
        if "latents" in batch:
            latents_cpu = batch["latents"]
            cpu_max = latents_cpu.abs().max().item()
            print(f"Step 3: Latents on CPU:")
            print(f"  Shape: {latents_cpu.shape}, dtype: {latents_cpu.dtype}, device: {latents_cpu.device}")
            print(f"  Max: {cpu_max:.6e}")
            
            if cpu_max < 1e-6:
                print("  ❌ FAILED: Latents are zeros on CPU!")
                return False
            else:
                print("  ✅ PASSED: Latents have valid values on CPU")
        
        # Step 4: Move to GPU (same as training - this is where it might fail)
        print(f"\nStep 4: Moving latents to GPU ({device})...")
        
        # Try the exact method used in training
        latents_gpu = latents_cpu.to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        
        gpu_max = latents_gpu.abs().max().item()
        print(f"  Latents on GPU:")
        print(f"    Shape: {latents_gpu.shape}, dtype: {latents_gpu.dtype}, device: {latents_gpu.device}")
        print(f"    Max: {gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Latents became zeros when moved to GPU!")
            print("  This is the exact point of failure in the training loop.")
            return False
        else:
            print("  ✅ PASSED: Latents preserved when moved to GPU")
        
        # Step 5: Generate noise (same as training)
        print(f"\nStep 5: Generating noise with torch.randn_like...")
        noise = torch.randn_like(latents_gpu)
        noise_max = noise.abs().max().item()
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        
        print(f"  Noise:")
        print(f"    Shape: {noise.shape}, dtype: {noise.dtype}, device: {noise.device}")
        print(f"    Max: {noise_max:.6e}, Mean: {noise_mean:.6e}, Std: {noise_std:.6e}")
        
        if noise_max < 1e-6:
            print("  ❌ FAILED: Noise is all zeros!")
            return False
        else:
            print("  ✅ PASSED: Noise generated correctly")
        
        # Step 6: Test with Accelerate (if used in training)
        print(f"\nStep 6: Testing with Accelerate...")
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        # Move latents using accelerator device
        latents_accel = latents_cpu.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        accel_max = latents_accel.abs().max().item()
        
        print(f"  Latents moved via Accelerator:")
        print(f"    Device: {latents_accel.device}, Max: {accel_max:.6e}")
        
        if accel_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Latents became zeros when moved via Accelerator!")
            print("  This indicates Accelerate is causing the issue.")
            return False
        else:
            print("  ✅ PASSED: Latents preserved when moved via Accelerator")
        
        # Step 7: Test multiple batches (simulate training loop)
        print(f"\nStep 7: Testing multiple batches (simulating training loop)...")
        batch_count = 0
        all_passed = True
        
        for i, batch in enumerate(train_dataloader):
            if i >= 3:  # Test first 3 batches
                break
            
            batch_count += 1
            if "latents" in batch:
                latents_batch = batch["latents"]
                cpu_max_batch = latents_batch.abs().max().item()
                
                latents_gpu_batch = latents_batch.to(device, non_blocking=False)
                torch.cuda.synchronize(device)
                gpu_max_batch = latents_gpu_batch.abs().max().item()
                
                print(f"  Batch {i+1}:")
                print(f"    CPU max: {cpu_max_batch:.6e}, GPU max: {gpu_max_batch:.6e}")
                
                if gpu_max_batch < 1e-6 and cpu_max_batch > 1e-6:
                    print(f"    ❌ FAILED: Batch {i+1} became zeros on GPU!")
                    all_passed = False
                else:
                    print(f"    ✅ PASSED: Batch {i+1} preserved")
        
        if not all_passed:
            return False
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL STEPS PASSED")
        print("\nThe training loop sequence works correctly in isolation.")
        print("If training still fails, the issue is likely:")
        print("  1. Model forward pass corrupting tensors")
        print("  2. Optimizer/backward pass issues")
        print("  3. Mixed precision (fp16) causing issues")
        print("  4. Gradient checkpointing causing issues")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_loop_sequence()
    sys.exit(0 if success else 1)


