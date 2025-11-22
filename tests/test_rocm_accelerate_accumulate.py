#!/usr/bin/env python3
"""
Test if accelerator.accumulate() causes tensor corruption.

This replicates the exact training sequence including accelerator.accumulate().
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_accelerate_accumulate():
    """Test if accelerator.accumulate() causes issues"""
    print("=" * 80)
    print("ROCm Accelerate Accumulate Test")
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
        
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16 * 64 * 64, 16 * 64 * 64)
            
            def forward(self, x):
                x = x.flatten(1)
                return self.linear(x).view(x.shape[0], 16, 64, 64)
        
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer = accelerator.prepare(model, optimizer)
        
        # Test: Transfer before accumulate
        print("Test 1: Transfer before accelerator.accumulate()")
        batch = next(iter(train_dataloader))
        latents = batch["latents"]
        cpu_max = latents.abs().max().item()
        
        if not latents.is_contiguous():
            latents = latents.contiguous()
        latents_clone = latents.clone()
        
        latents_gpu = latents_clone.pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_before = latents_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max:.6e}, GPU max before accumulate: {gpu_max_before:.6e}")
        if gpu_max_before < 1e-6 and cpu_max > 1e-6:
            print("  [FAILED] Transfer failed before accumulate")
            return False
        print("  [PASSED]")
        
        # Test: Inside accumulate context
        print("\nTest 2: Inside accelerator.accumulate() context")
        with accelerator.accumulate(model):
            # Check if latents are still valid
            gpu_max_inside = latents_gpu.abs().max().item()
            print(f"  GPU max inside accumulate: {gpu_max_inside:.6e}")
            if gpu_max_inside < 1e-6:
                print("  [FAILED] Latents became zeros inside accumulate context")
                return False
            
            # Forward pass
            output = model(latents_gpu)
            output_max = output.abs().max().item()
            print(f"  Output max: {output_max:.6e}")
            if output_max < 1e-6:
                print("  [FAILED] Output is zeros")
                return False
            
            # Backward pass
            target = torch.randn_like(output)
            loss = nn.functional.mse_loss(output, target)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Loss: {loss.item():.6e}")
        
        print("  [PASSED]")
        
        # Test: Transfer after accumulate
        print("\nTest 3: Transfer after accelerator.accumulate()")
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max_2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_clone = latents2.clone()
        
        latents2_gpu = latents2_clone.pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_after = latents2_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max_2:.6e}, GPU max after accumulate: {gpu_max_after:.6e}")
        if gpu_max_after < 1e-6 and cpu_max_2 > 1e-6:
            print("  [FAILED] Transfer failed after accumulate")
            print("  This indicates accelerator.accumulate() corrupts GPU state")
            return False
        print("  [PASSED]")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("[PASS] accelerator.accumulate() does not cause transfer failures")
        print("The issue must be in the WAN model forward pass or specific training operations")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_accelerate_accumulate()
    sys.exit(0 if success else 1)

