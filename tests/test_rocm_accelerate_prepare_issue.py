#!/usr/bin/env python3
"""
Test if Accelerate's prepare() method corrupts GPU state and causes transfer failures.

This test:
1. Sets up training context
2. Loads model
3. Prepares model/optimizer with Accelerate (like training does)
4. Tests if tensor transfers still work after prepare()

Run with: python tests/test_rocm_accelerate_prepare_issue.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_after_accelerate_prepare():
    """Test if Accelerate prepare() corrupts GPU state"""
    print("=" * 80)
    print("ROCm Accelerate prepare() Issue Test")
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
        from accelerate.utils import set_seed
        
        # Create accelerator (same as training)
        accelerator = Accelerator()
        print(f"Step 1: Created Accelerator, device: {accelerator.device}")
        
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
        
        # Get batch BEFORE prepare
        print("\nStep 2: Getting batch BEFORE Accelerate prepare()...")
        batch_before = next(iter(train_dataloader))
        latents_before = batch_before["latents"]
        cpu_max_before = latents_before.abs().max().item()
        print(f"  Latents max: {cpu_max_before:.6e}")
        
        # Test transfer BEFORE prepare
        if not latents_before.is_contiguous():
            latents_before = latents_before.contiguous()
        latents_before_clone = latents_before.clone()
        latents_before_gpu = latents_before_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_before = latents_before_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_before:.6e}")
        
        if gpu_max_before < 1e-6 and cpu_max_before > 1e-6:
            print("  ❌ FAILED: Transfer fails even BEFORE prepare()!")
            return False
        else:
            print("  ✅ PASSED: Transfer works before prepare()")
        
        # Now load and prepare model (same as training)
        print("\nStep 3: Loading model...")
        
        model_path = Path("models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors")
        if not model_path.exists():
            print(f"  ⚠️  Model not found at {model_path}")
            print("  Creating a dummy model for testing...")
            
            # Create a simple dummy model instead
            class DummyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = torch.nn.Linear(10, 10)
                
                def forward(self, x):
                    return self.layer(x)
            
            model = DummyModel()
            print("  Created dummy model")
        else:
            try:
                from musubi_tuner.wan.modules.model import WanModel
                print("  Loading DiT model (this may take a moment)...")
                model = WanModel.load_from_file(
                    str(model_path),
                    device="cpu",  # Load to CPU first, like training
                    dtype=torch.float16,
                    i2v=True,
                    flf2v=False,
                    v2_2=True
                )
                print("  Model loaded to CPU")
            except Exception as e:
                print(f"  ⚠️  Model loading failed: {e}")
                print("  Creating dummy model instead...")
                class DummyModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layer = torch.nn.Linear(10, 10)
                    def forward(self, x):
                        return self.layer(x)
                model = DummyModel()
        
        # Create optimizer (same as training)
        print("\nStep 4: Creating optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("  Optimizer created")
        
        # Create lr_scheduler (same as training)
        print("\nStep 5: Creating lr_scheduler...")
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        print("  LR scheduler created")
        
        # NOW prepare with Accelerate (this is the critical step)
        print("\nStep 6: Preparing model/optimizer/scheduler with Accelerate...")
        print("  This is where the issue might occur...")
        
        # Prepare model (same as training)
        if hasattr(model, 'move_to_device_except_swap_blocks'):
            # For real models with block swapping
            model = accelerator.prepare(model, device_placement=[False])
            model.move_to_device_except_swap_blocks(accelerator.device)
        else:
            # For dummy models
            model = accelerator.prepare(model)
        
        # Prepare optimizer and scheduler (same as training)
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        
        print("  Model/optimizer/scheduler prepared")
        print(f"  Model device: {next(model.parameters()).device}")
        
        # Clear cache after prepare
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        
        # Now test transfer AFTER prepare
        print("\nStep 7: Testing transfer AFTER Accelerate prepare()...")
        
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
            print("  ❌ FAILED: Transfer produces zeros AFTER Accelerate prepare()!")
            print("  This confirms Accelerate prepare() corrupts GPU state!")
            
            # Diagnostic: Can simple transfers work?
            print(f"\n  Diagnostic: Testing simple tensor transfer...")
            simple = torch.randn(100, device='cpu')
            simple_gpu = simple.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            simple_max = simple_gpu.abs().max().item()
            print(f"    Simple transfer: max={simple_max:.6e}")
            
            if simple_max > 1e-6:
                print("    ✅ Simple transfers work - issue is with this specific tensor")
                print("    This suggests Accelerate prepare() changes GPU memory state")
                print("    in a way that breaks transfers for certain tensor types/sizes")
            else:
                print("    ❌ Even simple transfers fail - GPU state is completely corrupted!")
            
            return False
        else:
            print("  ✅ PASSED: Transfer works even after Accelerate prepare()")
        
        # Test multiple batches
        print("\nStep 8: Testing multiple batches after prepare()...")
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
            
            if gpu_max < 1e-6 and cpu_max > 1e-6:
                print(f"  Batch {i+1}: ❌ FAILED (CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e})")
            else:
                print(f"  Batch {i+1}: ✅ PASSED (CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e})")
                success_count += 1
        
        print(f"\n  Summary: {success_count}/3 batches transferred successfully")
        
        if success_count < 3:
            print("  ⚠️  Some batches fail - this suggests an intermittent issue")
            return False
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nAccelerate prepare() does not corrupt GPU state.")
        print("The issue must be in the training loop itself.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_after_accelerate_prepare()
    sys.exit(0 if success else 1)


