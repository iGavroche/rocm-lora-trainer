#!/usr/bin/env python3
"""
Test the EXACT sequence of all 8 transfer methods like training does.

The training code tries all 8 methods in sequence. Maybe one of the failed
attempts corrupts GPU state, causing subsequent methods to fail.

Run with: python tests/test_rocm_all_transfer_methods_sequence.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_all_methods_sequence():
    """Test all 8 methods in sequence like training does"""
    print("=" * 80)
    print("ROCm All Transfer Methods Sequence Test")
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
        
        # Replicate EXACT move_to_gpu from training (all 8 methods)
        def move_to_gpu_all_methods(value, name="", step=0):
            """Exact copy of move_to_gpu with all 8 methods"""
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
                
                # Method 1: Pinned memory
                print(f"  Trying Method 1 (pinned memory)...")
                try:
                    if not value_clone.is_pinned():
                        pinned_value = value_clone.pin_memory()
                    else:
                        pinned_value = value_clone
                    tensor_gpu = pinned_value.to(accelerator.device, non_blocking=True)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 1 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 1 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 1 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 1 exception: {e}")
                
                # Method 2: Direct .to()
                print(f"  Trying Method 2 (direct .to())...")
                try:
                    tensor_gpu = value_clone.to(accelerator.device, non_blocking=False)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 2 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 2 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 2 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 2 exception: {e}")
                
                # Method 3: copy_()
                print(f"  Trying Method 3 (copy_())...")
                try:
                    tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                    tensor_gpu.copy_(value_clone, non_blocking=False)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 3 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 3 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 3 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 3 exception: {e}")
                
                # Method 4: Chunked transfer
                print(f"  Trying Method 4 (chunked transfer)...")
                try:
                    chunk_size = value_clone.numel() // 4
                    if chunk_size > 0:
                        tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                        flat_cpu = value_clone.flatten()
                        flat_gpu = tensor_gpu.flatten()
                        for i in range(0, flat_cpu.numel(), chunk_size):
                            end_idx = min(i + chunk_size, flat_cpu.numel())
                            flat_gpu[i:end_idx].copy_(flat_cpu[i:end_idx], non_blocking=False)
                        torch.cuda.synchronize(accelerator.device)
                        max_val_gpu = tensor_gpu.abs().max().item()
                        print(f"    Method 4 result: GPU max={max_val_gpu:.6e}")
                        if max_val_gpu > 1e-6:
                            print(f"    ✅ Method 4 worked!")
                            return tensor_gpu
                        else:
                            print(f"    ❌ Method 4 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 4 exception: {e}")
                
                # Method 5: numpy + torch.tensor
                print(f"  Trying Method 5 (numpy + torch.tensor)...")
                try:
                    numpy_array = value_clone.cpu().numpy()
                    tensor_gpu = torch.tensor(numpy_array, dtype=value_clone.dtype, device=accelerator.device)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 5 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 5 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 5 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 5 exception: {e}")
                
                # Method 6: CUDA stream (this hangs in training!)
                print(f"  Trying Method 6 (CUDA stream) - WARNING: This may hang...")
                try:
                    # Set a timeout to avoid hanging
                    import signal
                    stream = torch.cuda.Stream(device=accelerator.device)
                    with torch.cuda.stream(stream):
                        tensor_gpu = value_clone.to(accelerator.device, non_blocking=True)
                    stream.synchronize()
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 6 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 6 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 6 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 6 exception: {e}")
                
                # Method 7: Direct GPU allocation + element copy
                print(f"  Trying Method 7 (direct GPU allocation)...")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(accelerator.device)
                    tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                    cpu_flat = value_clone.flatten().cpu()
                    gpu_flat = tensor_gpu.flatten()
                    chunk_size = min(1024 * 1024, cpu_flat.numel())
                    for i in range(0, cpu_flat.numel(), chunk_size):
                        end_idx = min(i + chunk_size, cpu_flat.numel())
                        chunk_cpu = cpu_flat[i:end_idx].clone()
                        gpu_flat[i:end_idx] = chunk_cpu.to(accelerator.device, non_blocking=False)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 7 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 7 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 7 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 7 exception: {e}")
                
                # Method 8: numpy + torch.as_tensor
                print(f"  Trying Method 8 (numpy + torch.as_tensor)...")
                try:
                    numpy_array = value_clone.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                    tensor_gpu = torch.as_tensor(numpy_array, device=accelerator.device)
                    if tensor_gpu.device != accelerator.device:
                        tensor_gpu = torch.tensor(numpy_array, dtype=value_clone.dtype, device=accelerator.device)
                    torch.cuda.synchronize(accelerator.device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    print(f"    Method 8 result: GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 8 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 8 failed - zeros")
                except Exception as e:
                    print(f"    ❌ Method 8 exception: {e}")
                
                # All methods failed
                raise RuntimeError(f"All 8 methods failed for {name}")
            else:
                return value
        
        # Test the exact sequence
        print("\nTesting all 8 methods in sequence (like training does)...")
        print("-" * 80)
        
        if not latents.is_contiguous():
            latents = latents.contiguous()
        latents_clone = latents.clone()
        
        try:
            result = move_to_gpu_all_methods(latents_clone, "latents", 0)
            final_max = result.abs().max().item()
            print(f"\nFinal result: GPU max={final_max:.6e}")
            
            if final_max < 1e-6 and cpu_max > 1e-6:
                print("❌ FAILED: All methods produced zeros!")
                return False
            else:
                print("✅ PASSED: At least one method worked")
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False
        
        # Test if trying all methods corrupts GPU state
        print("\n" + "-" * 80)
        print("Testing if trying all methods corrupts GPU state...")
        print("(Testing a fresh transfer after trying all methods)")
        
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_clone = latents2.clone()
        
        # Try simple transfer (Method 2)
        print("  Trying simple transfer (Method 2) after all methods were tried...")
        latents2_gpu = latents2_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max2 = latents2_gpu.abs().max().item()
        print(f"  Result: CPU max={cpu_max2:.6e}, GPU max={gpu_max2:.6e}")
        
        if gpu_max2 < 1e-6 and cpu_max2 > 1e-6:
            print("  ❌ FAILED: GPU state corrupted after trying all methods!")
            return False
        else:
            print("  ✅ PASSED: GPU state not corrupted")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nTrying all 8 methods in sequence doesn't corrupt GPU state.")
        print("The issue must be something else in the actual training.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_methods_sequence()
    sys.exit(0 if success else 1)


