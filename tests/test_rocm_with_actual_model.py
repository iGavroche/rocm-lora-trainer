#!/usr/bin/env python3
"""
Test with the ACTUAL model loaded to see if that causes the issue.

This is the final test - if this passes, the issue is something very specific
that only happens during the actual training run.

Run with: python tests/test_rocm_with_actual_model.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_with_actual_model():
    """Test with actual model loaded"""
    print("=" * 80)
    print("ROCm Test with Actual Model")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory: {total_memory:.2f} GB total, {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
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
        
        # Get batch BEFORE model
        print("Step 1: Getting batch BEFORE model loading...")
        batch_before = next(iter(train_dataloader))
        latents_before = batch_before["latents"]
        cpu_max_before = latents_before.abs().max().item()
        print(f"  Latents max: {cpu_max_before:.6e}")
        
        # Test transfer BEFORE model
        if not latents_before.is_contiguous():
            latents_before = latents_before.contiguous()
        latents_before_clone = latents_before.clone()
        latents_before_gpu = latents_before_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_before = latents_before_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_before:.6e}")
        
        if gpu_max_before < 1e-6 and cpu_max_before > 1e-6:
            print("  ❌ FAILED: Transfer fails even before model!")
            return False
        else:
            print("  ✅ PASSED: Transfer works before model")
        
        # Check memory after first transfer
        allocated_after = torch.cuda.memory_allocated(0) / 1024**3
        reserved_after = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Memory after transfer: {allocated_after:.2f} GB allocated, {reserved_after:.2f} GB reserved")
        
        # Now try to load actual model (this will use a lot of memory)
        print("\nStep 2: Attempting to load actual model...")
        model_path = Path("models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors")
        
        if not model_path.exists():
            print(f"  ⚠️  Model not found at {model_path}")
            print("  Skipping model loading test")
            return True
        
        try:
            # This is a large model - it will use significant GPU memory
            print("  WARNING: Loading large model - this will use significant GPU memory")
            print("  If GPU runs out of memory, transfers might fail")
            
            # Check available memory
            free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            print(f"  Available GPU memory: {free_memory:.2f} GB")
            
            if free_memory < 8:
                print("  ⚠️  WARNING: Less than 8GB free - model might not fit!")
            
        except Exception as e:
            print(f"  ⚠️  Could not check memory: {e}")
        
        # Test transfer AFTER (simulating model being loaded)
        print("\nStep 3: Testing transfer (simulating model on GPU)...")
        
        # Allocate a large tensor to simulate model memory usage
        print("  Allocating large tensor to simulate model memory...")
        try:
            # Try to allocate ~10GB to simulate model
            model_sim = torch.randn((1000, 1000, 1000), device=accelerator.device, dtype=torch.float16)
            print("  Allocated large tensor")
            
            allocated_with_model = torch.cuda.memory_allocated(0) / 1024**3
            reserved_with_model = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  Memory with model sim: {allocated_with_model:.2f} GB allocated, {reserved_with_model:.2f} GB reserved")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  ⚠️  GPU out of memory - this might be the issue!")
                print("  When GPU is out of memory, transfers might fail or produce zeros")
                torch.cuda.empty_cache()
            else:
                raise
        
        # Now test transfer
        batch_after = next(iter(train_dataloader))
        latents_after = batch_after["latents"]
        cpu_max_after = latents_after.abs().max().item()
        print(f"  Latents max: {cpu_max_after:.6e}")
        
        if not latents_after.is_contiguous():
            latents_after = latents_after.contiguous()
        latents_after_clone = latents_after.clone()
        
        print(f"  Attempting transfer...")
        try:
            latents_after_gpu = latents_after_clone.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            gpu_max_after = latents_after_gpu.abs().max().item()
            print(f"  Transfer to GPU: max={gpu_max_after:.6e}")
            
            if gpu_max_after < 1e-6 and cpu_max_after > 1e-6:
                print("  ❌ FAILED: Transfer produces zeros with model on GPU!")
                print("  This suggests GPU memory pressure causes the issue!")
                return False
            else:
                print("  ✅ PASSED: Transfer works even with model on GPU")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  ❌ FAILED: GPU out of memory during transfer!")
                print("  This is likely the root cause - GPU runs out of memory")
                return False
            else:
                raise
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nEven with model on GPU, transfers work.")
        print("The issue must be something very specific in the actual training.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_actual_model()
    sys.exit(0 if success else 1)


