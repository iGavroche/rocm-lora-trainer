#!/usr/bin/env python3
"""
Test if the training step (forward/backward pass) causes the transfer issue.

This test:
1. Sets up everything like training
2. Performs a forward pass
3. Tests if tensor transfers still work after forward pass

Run with: python tests/test_rocm_training_step_issue.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_after_training_step():
    """Test if training step corrupts GPU state"""
    print("=" * 80)
    print("ROCm Training Step Issue Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Setup (same as training)
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
        print("Step 1: Getting batch...")
        batch = next(iter(train_dataloader))
        latents = batch["latents"]
        cpu_max = latents.abs().max().item()
        print(f"  Latents max: {cpu_max:.6e}")
        
        # Test transfer BEFORE training step
        print("\nStep 2: Testing transfer BEFORE training step...")
        if not latents.is_contiguous():
            latents = latents.contiguous()
        latents_clone = latents.clone()
        latents_gpu = latents_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_before = latents_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_before:.6e}")
        
        if gpu_max_before < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Transfer fails even before training step!")
            return False
        else:
            print("  ✅ PASSED: Transfer works before training step")
        
        # Create a simple model for testing
        print("\nStep 3: Creating simple model for forward pass...")
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(65536, 1024)  # Match latents shape
                self.layer2 = torch.nn.Linear(1024, 65536)
            
            def forward(self, x):
                # Flatten, process, reshape
                b, c, f, h, w = x.shape
                x_flat = x.flatten(1)
                x = self.layer1(x_flat)
                x = torch.relu(x)
                x = self.layer2(x)
                x = x.view(b, c, f, h, w)
                return x
        
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Prepare with Accelerate
        model, optimizer = accelerator.prepare(model, optimizer)
        print("  Model and optimizer prepared")
        
        # Perform a forward pass (simulating training)
        print("\nStep 4: Performing forward pass...")
        model.train()
        optimizer.zero_grad()
        
        # Use the transferred latents
        latents_gpu.requires_grad_(True)
        output = model(latents_gpu)
        loss = output.mean()
        
        print(f"  Forward pass completed, loss: {loss.item():.6e}")
        
        # Perform backward pass
        print("\nStep 5: Performing backward pass...")
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        print("  Backward pass completed")
        
        # Clear gradients and cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        
        # Now test transfer AFTER training step
        print("\nStep 6: Testing transfer AFTER training step...")
        
        # Get a fresh batch
        batch_after = next(iter(train_dataloader))
        latents_after = batch_after["latents"]
        cpu_max_after = latents_after.abs().max().item()
        print(f"  Latents max: {cpu_max_after:.6e}")
        
        # Test transfer
        if not latents_after.is_contiguous():
            latents_after = latents_after.contiguous()
        latents_after_clone = latents_after.clone()
        
        print(f"  Attempting transfer to {accelerator.device}...")
        latents_after_gpu = latents_after_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_after = latents_after_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_after:.6e}")
        
        if gpu_max_after < 1e-6 and cpu_max_after > 1e-6:
            print("  ❌ FAILED: Transfer produces zeros AFTER training step!")
            print("  This confirms the training step corrupts GPU state!")
            
            # Diagnostic: Can simple transfers work?
            print(f"\n  Diagnostic: Testing simple tensor transfer...")
            simple = torch.randn(100, device='cpu')
            simple_gpu = simple.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            simple_max = simple_gpu.abs().max().item()
            print(f"    Simple transfer: max={simple_max:.6e}")
            
            if simple_max > 1e-6:
                print("    ✅ Simple transfers work - issue is with this specific tensor")
                print("    This suggests the forward/backward pass changes GPU memory state")
                print("    in a way that breaks transfers for certain tensor types/sizes")
            else:
                print("    ❌ Even simple transfers fail - GPU state is completely corrupted!")
            
            return False
        else:
            print("  ✅ PASSED: Transfer works even after training step")
        
        # Test multiple training steps
        print("\nStep 7: Testing multiple training steps...")
        success_count = 0
        for i in range(3):
            batch = next(iter(train_dataloader))
            latents = batch["latents"]
            if not latents.is_contiguous():
                latents = latents.contiguous()
            latents_clone = latents.clone()
            latents_gpu = latents_clone.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            gpu_max = latents_gpu.abs().max().item()
            cpu_max = latents.abs().max().item()
            
            # Perform forward/backward
            optimizer.zero_grad()
            latents_gpu.requires_grad_(True)
            output = model(latents_gpu)
            loss = output.mean()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize(accelerator.device)
            
            # Get next batch and test transfer
            batch_next = next(iter(train_dataloader))
            latents_next = batch_next["latents"]
            if not latents_next.is_contiguous():
                latents_next = latents_next.contiguous()
            latents_next_clone = latents_next.clone()
            latents_next_gpu = latents_next_clone.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            gpu_max_next = latents_next_gpu.abs().max().item()
            cpu_max_next = latents_next.abs().max().item()
            
            if gpu_max_next < 1e-6 and cpu_max_next > 1e-6:
                print(f"  Step {i+1}: ❌ FAILED (CPU max={cpu_max_next:.6e}, GPU max={gpu_max_next:.6e})")
            else:
                print(f"  Step {i+1}: ✅ PASSED (CPU max={cpu_max_next:.6e}, GPU max={gpu_max_next:.6e})")
                success_count += 1
        
        print(f"\n  Summary: {success_count}/3 steps transferred successfully")
        
        if success_count < 3:
            print("  ⚠️  Some steps fail - this suggests the issue occurs during training")
            return False
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nTraining step does not corrupt GPU state.")
        print("The issue must be something very specific in the actual training code.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_after_training_step()
    sys.exit(0 if success else 1)


