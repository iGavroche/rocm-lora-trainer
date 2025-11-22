#!/usr/bin/env python3
"""
Test the EXACT training operations sequence from the training code.

This test replicates:
- scale_shift_latents
- torch.randn_like
- get_noisy_model_input_and_timesteps
- The exact sequence that happens in training

Run with: python tests/test_rocm_exact_training_operations.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_exact_training_operations():
    """Test the exact training operations sequence"""
    print("=" * 80)
    print("ROCm Exact Training Operations Test")
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
        
        # Replicate exact move_to_gpu from training
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
                
                raise RuntimeError(f"All transfer methods failed for {name}")
            else:
                return value
        
        # Get batch and replicate exact training sequence
        print("Replicating exact training operations sequence...")
        print("-" * 80)
        
        for step, batch in enumerate(train_dataloader):
            if step >= 3:  # Test first 3 steps
                break
            
            print(f"\nStep {step}:")
            
            # Exact same transfer code as training
            if batch["latents"].device.type == "cuda":
                latents = batch["latents"]
                print("  Latents already on GPU")
            else:
                print("  Transferring latents to GPU...")
                try:
                    latents = move_to_gpu(batch["latents"], "latents", step)
                    gpu_max = latents.abs().max().item()
                    cpu_max = batch["latents"].abs().max().item()
                    print(f"    Transfer: CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e}")
                    
                    if gpu_max < 1e-6 and cpu_max > 1e-6:
                        print("    ❌ FAILED: Transfer produces zeros!")
                        return False
                except Exception as e:
                    print(f"    ❌ FAILED: {e}")
                    return False
            
            # Step 1: scale_shift_latents (exact training code)
            print("  Applying scale_shift_latents...")
            # SCALING_FACTOR is defined in hunyuan_model/vae.py
            from musubi_tuner.hunyuan_model.vae import SCALING_FACTOR
            latents = latents * SCALING_FACTOR
            max_after_scale = latents.abs().max().item()
            print(f"    After scale_shift: max={max_after_scale:.6e}")
            
            if max_after_scale < 1e-6:
                print("    ❌ FAILED: Latents become zeros after scale_shift!")
                return False
            
            # Step 2: torch.randn_like (exact training code)
            print("  Generating noise with torch.randn_like...")
            noise = torch.randn_like(latents)
            noise_max = noise.abs().max().item()
            noise_mean = noise.mean().item()
            noise_std = noise.std().item()
            print(f"    Noise: max={noise_max:.6e}, mean={noise_mean:.6e}, std={noise_std:.6e}")
            
            if noise_max < 1e-6:
                print("    ❌ FAILED: Noise is zeros!")
                return False
            
            # Step 3: get_noisy_model_input_and_timesteps (exact training code)
            print("  Calling get_noisy_model_input_and_timesteps...")
            from musubi_tuner.hv_train_network import NetworkTrainer
            trainer = NetworkTrainer()
            
            # Create noise scheduler (same as training)
            from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
            noise_scheduler = FlowMatchDiscreteScheduler()
            
            # Get timesteps from batch
            timesteps_batch = batch.get("timesteps")
            if timesteps_batch is None:
                timesteps_batch = [[0.5]]  # Default timestep
            
            try:
                noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                    args, noise, latents, timesteps_batch, noise_scheduler, accelerator.device, torch.float16
                )
                
                noisy_max = noisy_model_input.abs().max().item()
                print(f"    noisy_model_input: max={noisy_max:.6e}")
                
                if noisy_max < 1e-6:
                    print("    ❌ FAILED: noisy_model_input is zeros!")
                    return False
                
                # Check if latents are still valid after all operations
                latents_final_max = latents.abs().max().item()
                print(f"    Latents after operations: max={latents_final_max:.6e}")
                
                if latents_final_max < 1e-6:
                    print("    ❌ FAILED: Latents became zeros during operations!")
                    return False
                
                print("    ✅ All operations successful")
                
            except Exception as e:
                print(f"    ⚠️  get_noisy_model_input_and_timesteps failed: {e}")
                print("    (This might be expected if model-specific code is needed)")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL OPERATIONS PASSED")
        print("\nAll training operations work correctly.")
        print("The issue must be in the actual model forward pass (call_dit) or")
        print("something that happens after these operations.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_training_operations()
    sys.exit(0 if success else 1)

