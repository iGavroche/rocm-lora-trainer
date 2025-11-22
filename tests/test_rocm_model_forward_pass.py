#!/usr/bin/env python3
"""
Test model forward pass to see if it corrupts tensors.

This tests if the model itself (forward pass, mixed precision, etc.) causes tensors to become zeros.
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_model_forward_pass():
    """Test if model forward pass corrupts tensors"""
    print("=" * 80)
    print("ROCm Model Forward Pass Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Load a batch (same as training)
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
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collator
        )
        
        batch = next(iter(train_dataloader))
        print("Step 1: Loaded batch from DataLoader")
        
        # Move latents to GPU
        latents = batch["latents"].to(device, non_blocking=False)
        latents_max_before = latents.abs().max().item()
        print(f"Step 2: Latents on GPU, max: {latents_max_before:.6e}")
        
        if latents_max_before < 1e-6:
            print("  ❌ FAILED: Latents are zeros before model forward pass!")
            return False
        
        # Test 1: Simple operations (addition, multiplication)
        print("\nTest 1: Simple tensor operations...")
        latents_add = latents + 1.0
        latents_mul = latents * 2.0
        add_max = latents_add.abs().max().item()
        mul_max = latents_mul.abs().max().item()
        print(f"  latents + 1.0: max={add_max:.6e}")
        print(f"  latents * 2.0: max={mul_max:.6e}")
        
        if add_max < 1e-6 or mul_max < 1e-6:
            print("  ❌ FAILED: Simple operations produce zeros!")
            return False
        else:
            print("  ✅ PASSED: Simple operations work")
        
        # Test 2: Mixed precision (fp16) operations
        print("\nTest 2: Mixed precision (fp16) operations...")
        latents_fp16 = latents.to(torch.float16)
        latents_fp16_max = latents_fp16.abs().max().item()
        print(f"  latents.to(float16): max={latents_fp16_max:.6e}")
        
        if latents_fp16_max < 1e-6 and latents_max_before > 1e-6:
            print("  ❌ FAILED: Conversion to fp16 produces zeros!")
            return False
        else:
            print("  ✅ PASSED: fp16 conversion works")
        
        # Test 3: Autocast context (like training uses)
        print("\nTest 3: Autocast context (fp16)...")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            latents_autocast = latents * 1.0  # Simple operation in autocast
            latents_autocast_max = latents_autocast.abs().max().item()
            print(f"  latents in autocast: max={latents_autocast_max:.6e}")
            
            if latents_autocast_max < 1e-6 and latents_max_before > 1e-6:
                print("  ❌ FAILED: Autocast produces zeros!")
                return False
            else:
                print("  ✅ PASSED: Autocast works")
        
        # Test 4: Check if latents are still valid after all operations
        print("\nTest 4: Verify latents still valid after operations...")
        latents_final_max = latents.abs().max().item()
        print(f"  Original latents max: {latents_final_max:.6e}")
        
        if latents_final_max < 1e-6:
            print("  ❌ FAILED: Latents became zeros during operations!")
            return False
        else:
            print("  ✅ PASSED: Latents preserved")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nModel forward pass operations work correctly.")
        print("The issue must be in:")
        print("  1. Specific model architecture operations")
        print("  2. Optimizer/backward pass")
        print("  3. Gradient checkpointing")
        print("  4. A specific sequence in the actual training code")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_forward_pass()
    sys.exit(0 if success else 1)


