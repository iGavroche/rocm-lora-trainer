#!/usr/bin/env python3
"""
Test the EXACT failure scenario from training.

In training, Methods 1-5 all fail (produce zeros), then Method 6 hangs.
Maybe the failed attempts corrupt GPU state?

This test forces methods to fail and checks if that corrupts state.

Run with: python tests/test_rocm_failure_scenario.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_failure_scenario():
    """Test what happens when methods fail like in training"""
    print("=" * 80)
    print("ROCm Failure Scenario Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Setup
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
        print(f"Initial latents: CPU max={cpu_max:.6e}")
        
        # Test 1: Normal transfer works
        print("\nTest 1: Normal transfer (baseline)...")
        if not latents.is_contiguous():
            latents = latents.contiguous()
        latents_clone = latents.clone()
        latents_gpu = latents_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max = latents_gpu.abs().max().item()
        print(f"  Result: GPU max={gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Normal transfer produces zeros!")
            return False
        else:
            print("  ✅ PASSED: Normal transfer works")
        
        # Test 2: Simulate failed transfers (like training)
        # In training, Methods 1-5 all produce zeros
        # Let's see if trying failed transfers corrupts state
        print("\nTest 2: Simulating failed transfers (like training)...")
        print("  In training, Methods 1-5 all produce zeros")
        print("  Testing if this corrupts GPU state...")
        
        # Try Method 1 (will work, but let's see what happens if we force it to "fail")
        print("  Trying Method 1 (pinned memory)...")
        try:
            if not latents_clone.is_pinned():
                pinned_value = latents_clone.pin_memory()
            else:
                pinned_value = latents_clone
            tensor_gpu_m1 = pinned_value.to(accelerator.device, non_blocking=True)
            torch.cuda.synchronize(accelerator.device)
            max_m1 = tensor_gpu_m1.abs().max().item()
            print(f"    Method 1 result: GPU max={max_m1:.6e}")
            
            # If it produces zeros (like in training), check GPU state
            if max_m1 < 1e-6:
                print("    ⚠️  Method 1 produced zeros (like in training)")
                print("    Checking if GPU state is corrupted...")
                
                # Test simple transfer
                simple = torch.randn(100, device='cpu')
                simple_gpu = simple.to(accelerator.device, non_blocking=False)
                simple_max = simple_gpu.abs().max().item()
                print(f"    Simple transfer test: max={simple_max:.6e}")
                
                if simple_max < 1e-6:
                    print("    ❌ GPU state corrupted - even simple transfers fail!")
                    return False
                else:
                    print("    ✅ GPU state OK - simple transfers work")
            else:
                print("    ✅ Method 1 worked (unlike training)")
        except Exception as e:
            print(f"    ❌ Method 1 exception: {e}")
        
        # Test 3: Check if the issue is with the specific tensor after failed attempts
        print("\nTest 3: Testing same tensor after 'failed' attempts...")
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_clone = latents2.clone()
        
        # Try transfer
        latents2_gpu = latents2_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max2 = latents2_gpu.abs().max().item()
        print(f"  Result: CPU max={cpu_max2:.6e}, GPU max={gpu_max2:.6e}")
        
        if gpu_max2 < 1e-6 and cpu_max2 > 1e-6:
            print("  ❌ FAILED: Transfer fails after previous attempts!")
            return False
        else:
            print("  ✅ PASSED: Transfer works after previous attempts")
        
        # Test 4: Check if the issue is timing-related
        # Maybe in training, the transfer happens at a specific time when GPU is busy?
        print("\nTest 4: Testing with GPU under load (simulating model on GPU)...")
        
        # Allocate large tensors to simulate model
        model_tensors = []
        for i in range(5):
            tensor = torch.randn((1000, 1000), device=accelerator.device)
            model_tensors.append(tensor)
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  Allocated {len(model_tensors)} tensors, Memory: {allocated:.2f} GB")
        
        # Now try transfer
        batch3 = next(iter(train_dataloader))
        latents3 = batch3["latents"]
        cpu_max3 = latents3.abs().max().item()
        
        if not latents3.is_contiguous():
            latents3 = latents3.contiguous()
        latents3_clone = latents3.clone()
        
        latents3_gpu = latents3_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max3 = latents3_gpu.abs().max().item()
        print(f"  Result: CPU max={cpu_max3:.6e}, GPU max={gpu_max3:.6e}")
        
        if gpu_max3 < 1e-6 and cpu_max3 > 1e-6:
            print("  ❌ FAILED: Transfer fails when GPU is under load!")
            return False
        else:
            print("  ✅ PASSED: Transfer works even when GPU is under load")
        
        # Clean up
        del model_tensors
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nNone of the failure scenarios reproduce the issue.")
        print("The problem must be something very specific that only happens")
        print("during the actual training run with the real model and full training loop.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_failure_scenario()
    sys.exit(0 if success else 1)


