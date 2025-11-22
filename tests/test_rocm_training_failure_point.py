#!/usr/bin/env python3
"""
Test to identify the exact point where tensors become zeros in training.

This test simulates the exact training sequence and checks tensor values
at each step to find where corruption occurs.
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_training_failure_point():
    """Test to find exact failure point"""
    print("=" * 80)
    print("ROCm Training Failure Point Test")
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
        
        # Check point 1: After DataLoader
        print("Check Point 1: After DataLoader")
        cpu_max_1 = latents.abs().max().item()
        print(f"  CPU max: {cpu_max_1:.6e}")
        if cpu_max_1 < 1e-6:
            print("  ❌ FAILED: Tensors are zeros after DataLoader!")
            return False
        print("  [PASSED]")
        
        # Check point 2: After making contiguous
        if not latents.is_contiguous():
            latents = latents.contiguous()
        print("\nCheck Point 2: After contiguous()")
        cpu_max_2 = latents.abs().max().item()
        print(f"  CPU max: {cpu_max_2:.6e}")
        if cpu_max_2 < 1e-6:
            print("  ❌ FAILED: Tensors are zeros after contiguous()!")
            return False
        print("  [PASSED]")
        
        # Check point 3: After clone
        latents_clone = latents.clone()
        print("\nCheck Point 3: After clone()")
        cpu_max_3 = latents_clone.abs().max().item()
        print(f"  CPU max: {cpu_max_3:.6e}")
        if cpu_max_3 < 1e-6:
            print("  ❌ FAILED: Tensors are zeros after clone()!")
            return False
        print("  [PASSED]")
        
        # Check point 4: After pin_memory
        try:
            pinned = latents_clone.pin_memory()
            print("\nCheck Point 4: After pin_memory()")
            cpu_max_4 = pinned.abs().max().item()
            print(f"  CPU max: {cpu_max_4:.6e}")
            if cpu_max_4 < 1e-6:
                print("  ❌ FAILED: Tensors are zeros after pin_memory()!")
                return False
            print("  [PASSED]")
        except Exception as e:
            print(f"  ⚠️  pin_memory() failed: {e}")
            pinned = latents_clone
        
        # Check point 5: After .to(device)
        print("\nCheck Point 5: After .to(device)")
        latents_gpu = pinned.to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_5 = latents_gpu.abs().max().item()
        print(f"  GPU max: {gpu_max_5:.6e}")
        if gpu_max_5 < 1e-6:
            print("  ❌ FAILED: Tensors are zeros after .to(device)!")
            print("  This is the failure point!")
            return False
        print("  [PASSED]")
        
        # Check point 6: After scale_shift_latents (simulate)
        print("\nCheck Point 6: After scale_shift_latents (simulated)")
        # Simple scale/shift simulation
        scaled = latents_gpu * 0.18215 + 0.0
        gpu_max_6 = scaled.abs().max().item()
        print(f"  GPU max: {gpu_max_6:.6e}")
        if gpu_max_6 < 1e-6:
            print("  ❌ FAILED: Tensors are zeros after scale_shift!")
            return False
        print("  [PASSED]")
        
        # Check point 7: After noise generation
        print("\nCheck Point 7: After noise generation")
        noise = torch.randn_like(scaled)
        noise_max = noise.abs().max().item()
        print(f"  Noise max: {noise_max:.6e}")
        if noise_max < 1e-6:
            print("  ❌ FAILED: Noise is zeros!")
            return False
        print("  [PASSED]")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("[PASS] All check points passed in isolation")
        print("The issue must occur in a different context (with model loaded, etc.)")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_failure_point()
    sys.exit(0 if success else 1)

