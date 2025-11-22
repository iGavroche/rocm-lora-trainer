#!/usr/bin/env python3
"""
Test if accelerator.prepare(transformer) corrupts GPU state.

This simulates the exact sequence: load model, prepare with Accelerate,
then test transfers.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_accelerate_prepare_transformer():
    """Test if prepare(transformer) causes issues"""
    print("=" * 80)
    print("ROCm Accelerate Prepare Transformer Test")
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
        
        # Test 1: Transfer before model
        print("Test 1: Transfer before model")
        batch1 = next(iter(train_dataloader))
        latents1 = batch1["latents"]
        cpu_max_1 = latents1.abs().max().item()
        
        if not latents1.is_contiguous():
            latents1 = latents1.contiguous()
        latents1_gpu = latents1.clone().pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_1 = latents1_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max_1:.6e}, GPU max: {gpu_max_1:.6e}")
        if gpu_max_1 < 1e-6 and cpu_max_1 > 1e-6:
            print("  [FAILED]")
            return False
        print("  [PASSED]")
        
        # Test 2: Create large model (simulate WAN model size)
        print("\nTest 2: Creating large model (simulating WAN model)")
        # Create a model that uses significant GPU memory
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use multiple large layers to simulate WAN model memory usage
                self.layers = nn.ModuleList([
                    nn.Linear(4096, 4096) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = LargeModel().to(accelerator.device)
        allocated_before_prepare = torch.cuda.memory_allocated(accelerator.device) / 1024**3
        print(f"  Model on GPU: {allocated_before_prepare:.2f} GB allocated")
        
        # Test 3: Prepare with Accelerate (like training does)
        print("\nTest 3: Preparing model with Accelerate")
        model = accelerator.prepare(model)
        allocated_after_prepare = torch.cuda.memory_allocated(accelerator.device) / 1024**3
        print(f"  After prepare: {allocated_after_prepare:.2f} GB allocated")
        
        # Test 4: Transfer after prepare
        print("\nTest 4: Transfer after accelerator.prepare(model)")
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max_2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_gpu = latents2.clone().pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_2 = latents2_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max_2:.6e}, GPU max: {gpu_max_2:.6e}")
        if gpu_max_2 < 1e-6 and cpu_max_2 > 1e-6:
            print("  [FAILED] Transfer fails after accelerator.prepare(model)!")
            print("  THIS IS THE ROOT CAUSE: accelerator.prepare() corrupts GPU state")
            return False
        print("  [PASSED]")
        
        print("\n[PASS] accelerator.prepare(model) does not cause transfer failures")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_accelerate_prepare_transformer()
    sys.exit(0 if success else 1)

