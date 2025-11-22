#!/usr/bin/env python3
"""
Test if forward/backward pass causes tensor corruption.

This simulates the actual training step to see if forward/backward pass
corrupts GPU state and causes subsequent transfers to fail.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_forward_backward_pass():
    """Test if forward/backward pass causes issues"""
    print("=" * 80)
    print("ROCm Forward/Backward Pass Test")
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
        
        # Create a simple model to simulate training
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16 * 64 * 64, 16 * 64 * 64)
            
            def forward(self, x):
                x = x.flatten(1)
                return self.linear(x).view(x.shape[0], 16, 64, 64)
        
        model = SimpleModel().to(accelerator.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer = accelerator.prepare(model, optimizer)
        
        # Test: Transfer before forward/backward
        print("Test 1: Transfer before forward/backward")
        batch1 = next(iter(train_dataloader))
        latents1 = batch1["latents"]
        cpu_max_1 = latents1.abs().max().item()
        
        if not latents1.is_contiguous():
            latents1 = latents1.contiguous()
        latents1_clone = latents1.clone()
        
        latents1_gpu = latents1_clone.pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_1 = latents1_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max_1:.6e}, GPU max: {gpu_max_1:.6e}")
        if gpu_max_1 < 1e-6 and cpu_max_1 > 1e-6:
            print("  [FAILED]")
            return False
        print("  [PASSED]")
        
        # Test: Forward pass
        print("\nTest 2: Forward pass")
        with accelerator.autocast():
            output = model(latents1_gpu)
        output_max = output.abs().max().item()
        print(f"  Output max: {output_max:.6e}")
        if output_max < 1e-6:
            print("  [FAILED] Output is zeros!")
            return False
        print("  [PASSED]")
        
        # Test: Backward pass
        print("\nTest 3: Backward pass")
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Loss: {loss.item():.6e}")
        print("  [PASSED]")
        
        # Test: Transfer after forward/backward
        print("\nTest 4: Transfer after forward/backward")
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max_2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_clone = latents2.clone()
        
        latents2_gpu = latents2_clone.pin_memory().to(accelerator.device, non_blocking=True)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_2 = latents2_gpu.abs().max().item()
        
        print(f"  CPU max: {cpu_max_2:.6e}, GPU max: {gpu_max_2:.6e}")
        if gpu_max_2 < 1e-6 and cpu_max_2 > 1e-6:
            print("  [FAILED] Transfer fails after forward/backward!")
            print("  This indicates forward/backward pass corrupts GPU state")
            return False
        print("  [PASSED]")
        
        # Test: Multiple forward/backward cycles
        print("\nTest 5: Multiple forward/backward cycles")
        for i in range(3):
            batch = next(iter(train_dataloader))
            latents = batch["latents"]
            cpu_max = latents.abs().max().item()
            
            if not latents.is_contiguous():
                latents = latents.contiguous()
            latents_clone = latents.clone()
            
            latents_gpu = latents_clone.pin_memory().to(accelerator.device, non_blocking=True)
            torch.cuda.synchronize(accelerator.device)
            gpu_max = latents_gpu.abs().max().item()
            
            if gpu_max < 1e-6 and cpu_max > 1e-6:
                print(f"  [FAILED] Cycle {i+1} - transfer failed")
                return False
            
            with accelerator.autocast():
                output = model(latents_gpu)
            target = torch.randn_like(output)
            loss = nn.functional.mse_loss(output, target)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        print("  [PASSED] All 3 cycles succeeded")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("[PASS] Forward/backward pass does not cause transfer failures")
        print("The issue must be specific to the WAN model or training code sequence")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forward_backward_pass()
    sys.exit(0 if success else 1)

