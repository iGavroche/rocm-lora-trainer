#!/usr/bin/env python3
"""
Test the exact move_to_gpu function used in training to see why it fails.

This replicates the exact function and tests it with the same tensor types/shapes.
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_move_to_gpu_function():
    """Test the exact move_to_gpu function from training code"""
    print("=" * 80)
    print("ROCm move_to_gpu Function Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    # Replicate the exact move_to_gpu function from training
    def move_to_gpu(value, name="", step=0):
        """Exact copy of move_to_gpu from hv_train_network.py"""
        # Handle non-tensor values (lists, None, etc.)
        if not isinstance(value, torch.Tensor):
            return value
        
        if value.device.type == "cpu":
            max_val_cpu = value.abs().max().item()
            if max_val_cpu < 1e-6:
                # Already zeros, just move it
                return value.to(device, non_blocking=False)
            
            # Method 1: Try with pinned memory (if available)
            try:
                print(f"  Trying Method 1 (pinned memory) for {name}...")
                # Pin the CPU tensor to page-locked memory first
                if not value.is_pinned():
                    pinned_value = value.pin_memory()
                else:
                    pinned_value = value
                
                tensor_gpu = pinned_value.to(device, non_blocking=True)
                # Synchronize to ensure transfer is complete
                torch.cuda.synchronize(device)
                max_val_gpu = tensor_gpu.abs().max().item()
                
                print(f"    Method 1 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                if max_val_gpu > 1e-6:
                    print(f"    ✅ Method 1 worked!")
                    return tensor_gpu
                else:
                    print(f"    ❌ Method 1 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 1 exception: {e}")
            
            # Method 2: Try direct .to() with non_blocking=False
            try:
                print(f"  Trying Method 2 (direct .to()) for {name}...")
                tensor_gpu = value.to(device, non_blocking=False)
                max_val_gpu = tensor_gpu.abs().max().item()
                
                print(f"    Method 2 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                if max_val_gpu > 1e-6:
                    print(f"    ✅ Method 2 worked!")
                    return tensor_gpu
                else:
                    print(f"    ❌ Method 2 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 2 exception: {e}")
            
            # Method 3: Try copy_() with empty tensor
            try:
                print(f"  Trying Method 3 (copy_()) for {name}...")
                tensor_gpu = torch.empty(value.shape, dtype=value.dtype, device=device)
                tensor_gpu.copy_(value, non_blocking=False)
                torch.cuda.synchronize(device)
                max_val_gpu = tensor_gpu.abs().max().item()
                
                print(f"    Method 3 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                if max_val_gpu > 1e-6:
                    print(f"    ✅ Method 3 worked!")
                    return tensor_gpu
                else:
                    print(f"    ❌ Method 3 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 3 exception: {e}")
            
            # Method 4: Try chunked transfer (for large tensors)
            try:
                print(f"  Trying Method 4 (chunked transfer) for {name}...")
                # Split into chunks and transfer separately
                chunk_size = value.numel() // 4  # 4 chunks
                if chunk_size > 0:
                    tensor_gpu = torch.empty(value.shape, dtype=value.dtype, device=device)
                    flat_cpu = value.flatten()
                    flat_gpu = tensor_gpu.flatten()
                    
                    for i in range(0, flat_cpu.numel(), chunk_size):
                        end_idx = min(i + chunk_size, flat_cpu.numel())
                        flat_gpu[i:end_idx].copy_(flat_cpu[i:end_idx], non_blocking=False)
                    
                    torch.cuda.synchronize(device)
                    max_val_gpu = tensor_gpu.abs().max().item()
                    
                    print(f"    Method 4 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                    if max_val_gpu > 1e-6:
                        print(f"    ✅ Method 4 worked!")
                        return tensor_gpu
                    else:
                        print(f"    ❌ Method 4 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 4 exception: {e}")
            
            # Method 5: Try using torch.tensor constructor on GPU
            try:
                print(f"  Trying Method 5 (tensor constructor) for {name}...")
                # Convert to numpy, then create tensor directly on GPU
                numpy_array = value.cpu().numpy()
                tensor_gpu = torch.tensor(numpy_array, dtype=value.dtype, device=device)
                torch.cuda.synchronize(device)
                max_val_gpu = tensor_gpu.abs().max().item()
                
                print(f"    Method 5 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                if max_val_gpu > 1e-6:
                    print(f"    ✅ Method 5 worked!")
                    return tensor_gpu
                else:
                    print(f"    ❌ Method 5 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 5 exception: {e}")
            
            # Method 6: Try using CUDA stream for async transfer
            try:
                print(f"  Trying Method 6 (CUDA stream) for {name}...")
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    tensor_gpu = value.to(device, non_blocking=True)
                stream.synchronize()
                max_val_gpu = tensor_gpu.abs().max().item()
                
                print(f"    Method 6 result: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                if max_val_gpu > 1e-6:
                    print(f"    ✅ Method 6 worked!")
                    return tensor_gpu
                else:
                    print(f"    ❌ Method 6 failed - GPU tensor is zeros")
            except Exception as e:
                print(f"    ❌ Method 6 exception: {e}")
            
            # All methods failed
            print(f"  ❌ ALL METHODS FAILED for {name}!")
            return None
        else:
            return value
    
    # Test 1: Simple tensor
    print("Test 1: Simple tensor")
    print("-" * 80)
    cpu_tensor = torch.randn((1, 16, 1, 64, 64), device='cpu')
    cpu_max = cpu_tensor.abs().max().item()
    print(f"CPU tensor max: {cpu_max:.6e}")
    
    gpu_tensor = move_to_gpu(cpu_tensor, "test_tensor", step=0)
    if gpu_tensor is None:
        print("❌ FAILED: All methods failed")
        return False
    
    gpu_max = gpu_tensor.abs().max().item()
    print(f"GPU tensor max: {gpu_max:.6e}")
    
    if gpu_max < 1e-6 and cpu_max > 1e-6:
        print("❌ FAILED: Tensor became zeros")
        return False
    else:
        print("✅ PASSED: Tensor preserved")
    
    # Test 2: Load actual batch from dataset
    print("\n" + "=" * 80)
    print("Test 2: Actual batch tensor from dataset")
    print("-" * 80)
    
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
    latents_batch = batch["latents"]
    
    print(f"Batch latents:")
    print(f"  Shape: {latents_batch.shape}, dtype: {latents_batch.dtype}, device: {latents_batch.device}")
    cpu_max_batch = latents_batch.abs().max().item()
    print(f"  CPU max: {cpu_max_batch:.6e}")
    
    print(f"\nTesting move_to_gpu on actual batch tensor...")
    gpu_tensor_batch = move_to_gpu(latents_batch, "latents", step=0)
    
    if gpu_tensor_batch is None:
        print("❌ FAILED: All methods failed on actual batch tensor!")
        print("This is the exact failure point in training.")
        return False
    
    gpu_max_batch = gpu_tensor_batch.abs().max().item()
    print(f"GPU tensor max: {gpu_max_batch:.6e}")
    
    if gpu_max_batch < 1e-6 and cpu_max_batch > 1e-6:
        print("❌ FAILED: Batch tensor became zeros!")
        return False
    else:
        print("✅ PASSED: Batch tensor preserved")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ move_to_gpu function works correctly")
    print("If training fails, the issue is likely:")
    print("  1. Different tensor state/properties in training")
    print("  2. Memory state during training")
    print("  3. Interaction with other operations")
    return True

if __name__ == "__main__":
    success = test_move_to_gpu_function()
    sys.exit(0 if success else 1)


