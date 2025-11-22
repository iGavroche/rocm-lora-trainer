#!/usr/bin/env python3
"""
Test the EXACT sequence from training code.

This replicates the exact check and transfer logic from hv_train_network.py
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_exact_training_sequence():
    """Test exact training sequence"""
    print("=" * 80)
    print("ROCm Exact Training Sequence Test")
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
        
        # Replicate exact training code logic
        def move_to_gpu(value, name=""):
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
                # Try pinned memory first
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
                # Fallback to direct .to()
                tensor_gpu = value_clone.to(accelerator.device, non_blocking=False)
                max_val_gpu = tensor_gpu.abs().max().item()
                if max_val_gpu > 1e-6:
                    return tensor_gpu
                raise RuntimeError(f"All transfer methods failed for {name}")
            else:
                return value
        
        # Get batch
        batch = next(iter(train_dataloader))
        
        # EXACT training code logic
        print("Step 1: Check batch device")
        print(f"  batch['latents'].device: {batch['latents'].device}")
        print(f"  batch['latents'].device.type: {batch['latents'].device.type}")
        print(f"  batch['latents'].max: {batch['latents'].abs().max().item():.6e}")
        
        if batch["latents"].device.type == "cuda":
            print("  [WARNING] Tensors already on GPU - this should not happen!")
            print("  Training code will skip transfer and use zeros")
            latents = batch["latents"]
            max_val = latents.abs().max().item()
            if max_val < 1e-6:
                print("  [FAILED] Tensors on GPU are zeros - this is the bug!")
                return False
        else:
            print("  [OK] Tensors on CPU - will transfer")
            latents = move_to_gpu(batch["latents"], "latents")
            max_val = latents.abs().max().item()
            print(f"  After transfer max: {max_val:.6e}")
            if max_val < 1e-6:
                print("  [FAILED] Transfer produced zeros!")
                return False
        
        print("\n[PASS] Exact training sequence works")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_training_sequence()
    sys.exit(0 if success else 1)
