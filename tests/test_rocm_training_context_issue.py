#!/usr/bin/env python3
"""
Test that simulates the EXACT training context to find what's different.

This test replicates:
- Accelerate setup
- DataLoader with collator
- Exact batch structure
- Model loading

Run with: python tests/test_rocm_training_context_issue.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_training_context():
    """Test the exact training context"""
    print("=" * 80)
    print("ROCm Training Context Issue Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print()
    
    try:
        # Step 1: Setup Accelerate (same as training)
        print("Step 1: Setting up Accelerate...")
        from accelerate import Accelerator
        accelerator = Accelerator()
        print(f"  Accelerator device: {accelerator.device}")
        
        # Step 2: Load dataset (same as training)
        print("\nStep 2: Loading dataset...")
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
        print(f"  Dataset: {type(dataset).__name__}, length: {len(dataset)}")
        
        # Step 3: Create DataLoader (same as training)
        print("\nStep 3: Creating DataLoader...")
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
        print("  DataLoader created (NOT prepared with Accelerate)")
        
        # Step 4: Get batch (same as training)
        print("\nStep 4: Getting batch from DataLoader...")
        batch = next(iter(train_dataloader))
        print(f"  Batch keys: {list(batch.keys())}")
        
        # Step 5: Check latents
        latents = batch["latents"]
        print(f"\nStep 5: Inspecting latents tensor...")
        print(f"  Shape: {latents.shape}")
        print(f"  Dtype: {latents.dtype}")
        print(f"  Device: {latents.device}")
        print(f"  Is contiguous: {latents.is_contiguous()}")
        print(f"  Storage type: {type(latents.storage())}")
        print(f"  Storage offset: {latents.storage_offset()}")
        print(f"  Max: {latents.abs().max().item():.6e}")
        print(f"  Mean: {latents.abs().mean().item():.6e}")
        
        cpu_max = latents.abs().max().item()
        if cpu_max < 1e-6:
            print("  ❌ FAILED: Latents are zeros on CPU!")
            return False
        
        # Step 6: Try transfer (same as training)
        print(f"\nStep 6: Transferring to GPU using accelerator.device ({accelerator.device})...")
        
        # Check if tensor has any special properties
        print(f"  Tensor properties:")
        print(f"    requires_grad: {latents.requires_grad}")
        print(f"    is_leaf: {latents.is_leaf}")
        print(f"    grad_fn: {latents.grad_fn}")
        
        # Make contiguous and clone (same as training)
        if not latents.is_contiguous():
            print("  Making contiguous...")
            latents = latents.contiguous()
        
        latents_clone = latents.clone()
        clone_max = latents_clone.abs().max().item()
        print(f"  After clone: max={clone_max:.6e}")
        
        if clone_max < 1e-6:
            print("  ❌ FAILED: Clone produces zeros!")
            return False
        
        # Try transfer
        print(f"  Attempting transfer...")
        latents_gpu = latents_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max = latents_gpu.abs().max().item()
        print(f"  GPU max: {gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("  ❌ FAILED: Transfer produces zeros!")
            print("  This is the exact issue from training!")
            
            # Try diagnostic: Can we transfer a simple tensor right now?
            print(f"\n  Diagnostic: Testing simple tensor transfer in this context...")
            simple = torch.randn(100, device='cpu')
            simple_gpu = simple.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            simple_max = simple_gpu.abs().max().item()
            print(f"    Simple tensor transfer: max={simple_max:.6e}")
            
            if simple_max > 1e-6:
                print("    ✅ Simple transfers work - issue is tensor-specific!")
                print("    This suggests the batch tensor has a special property.")
            else:
                print("    ❌ Even simple transfers fail - GPU state is corrupted!")
            
            return False
        else:
            print("  ✅ PASSED: Transfer successful")
        
        # Step 7: Test with model loaded (simulate full training)
        print(f"\nStep 7: Testing after model loading simulation...")
        
        # Allocate model-like tensors
        model_tensors = []
        for i in range(5):
            tensor = torch.randn((1000, 1000), device=accelerator.device)
            model_tensors.append(tensor)
        print(f"  Allocated {len(model_tensors)} model-like tensors")
        
        # Get another batch
        batch2 = next(iter(train_dataloader))
        latents2 = batch2["latents"]
        cpu_max2 = latents2.abs().max().item()
        
        if not latents2.is_contiguous():
            latents2 = latents2.contiguous()
        latents2_clone = latents2.clone()
        
        latents2_gpu = latents2_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max2 = latents2_gpu.abs().max().item()
        
        print(f"  Second batch transfer: CPU max={cpu_max2:.6e}, GPU max={gpu_max2:.6e}")
        
        if gpu_max2 < 1e-6 and cpu_max2 > 1e-6:
            print("  ❌ FAILED: Second batch transfer produces zeros!")
            return False
        else:
            print("  ✅ PASSED: Second batch transfer successful")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nTensors transfer correctly in training context.")
        print("If training still fails, the issue must be:")
        print("  1. Something that happens during the actual training loop")
        print("  2. Model forward pass corrupting tensors")
        print("  3. Optimizer/backward pass issues")
        print("  4. A specific sequence of operations")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_context()
    sys.exit(0 if success else 1)


