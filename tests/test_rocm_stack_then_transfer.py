#!/usr/bin/env python3
"""
Test if torch.stack() on CPU followed by GPU transfer causes zeros.

This tests the exact sequence: load to CPU, stack on CPU, then transfer to GPU.
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_stack_then_transfer():
    """Test stacking then transferring"""
    print("=" * 80)
    print("ROCm Stack Then Transfer Test")
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
        
        # Get batch (should be on CPU)
        batch = next(iter(train_dataloader))
        latents = batch["latents"]
        
        print(f"Step 1: After DataLoader")
        print(f"  Device: {latents.device}, max: {latents.abs().max().item():.6e}")
        
        if latents.device.type == "cuda":
            print("  [WARNING] Tensors are already on GPU from DataLoader!")
            print("  This should not happen - they should be on CPU")
        
        # Simulate the exact training sequence
        print(f"\nStep 2: Making contiguous")
        if not latents.is_contiguous():
            latents = latents.contiguous()
        max_after_contig = latents.abs().max().item()
        print(f"  Device: {latents.device}, max: {max_after_contig:.6e}")
        
        print(f"\nStep 3: Cloning")
        latents_clone = latents.clone()
        max_after_clone = latents_clone.abs().max().item()
        print(f"  Device: {latents_clone.device}, max: {max_after_clone:.6e}")
        
        print(f"\nStep 4: Pinning memory")
        pinned = latents_clone.pin_memory()
        max_after_pin = pinned.abs().max().item()
        print(f"  Device: {pinned.device}, max: {max_after_pin:.6e}")
        
        print(f"\nStep 5: Transferring to GPU")
        latents_gpu = pinned.to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        max_after_transfer = latents_gpu.abs().max().item()
        print(f"  Device: {latents_gpu.device}, max: {max_after_transfer:.6e}")
        
        if max_after_transfer < 1e-6 and max_after_pin > 1e-6:
            print("  [FAILED] Transfer produced zeros!")
            return False
        
        print("\n[PASS] Stack then transfer works correctly")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stack_then_transfer()
    sys.exit(0 if success else 1)

