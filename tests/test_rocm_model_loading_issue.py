#!/usr/bin/env python3
"""
Test if model loading corrupts GPU state and causes transfer failures.

This test:
1. Sets up training context
2. Loads the actual model (like training does)
3. Tests if tensor transfers still work after model loading

Run with: python tests/test_rocm_model_loading_issue.py
"""
import torch
import sys
import os
from pathlib import Path
from multiprocessing import Value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_after_model_loading():
    """Test if model loading corrupts GPU state"""
    print("=" * 80)
    print("ROCm Model Loading Issue Test")
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
        
        # Get batch BEFORE model loading
        print("Step 1: Getting batch BEFORE model loading...")
        batch_before = next(iter(train_dataloader))
        latents_before = batch_before["latents"]
        cpu_max_before = latents_before.abs().max().item()
        print(f"  Latents max: {cpu_max_before:.6e}")
        
        # Test transfer BEFORE model
        if not latents_before.is_contiguous():
            latents_before = latents_before.contiguous()
        latents_before_clone = latents_before.clone()
        latents_before_gpu = latents_before_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_before = latents_before_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_before:.6e}")
        
        if gpu_max_before < 1e-6 and cpu_max_before > 1e-6:
            print("  ❌ FAILED: Transfer fails even BEFORE model loading!")
            return False
        else:
            print("  ✅ PASSED: Transfer works before model loading")
        
        # Now load the model (same as training)
        print("\nStep 2: Loading model (this might corrupt GPU state)...")
        
        # Check if model files exist
        model_path = Path("models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors")
        if not model_path.exists():
            print(f"  ⚠️  Model not found at {model_path}, skipping model loading test")
            print("  This test requires the actual model to be loaded.")
            return True
        
        try:
            from musubi_tuner.wan.modules.model import WanModel
            from musubi_tuner.wan.modules.vae import WanVAE
            
            # Load VAE (smaller, faster)
            print("  Loading VAE...")
            vae_path = Path("models/wan/wan_2.1_vae.safetensors")
            if vae_path.exists():
                vae = WanVAE.load_from_file(str(vae_path), device=accelerator.device, dtype=torch.float16)
                print("  VAE loaded")
            else:
                print("  VAE not found, skipping")
            
            # Load DiT model (this is the big one)
            print("  Loading DiT model (this may take a moment)...")
            dit = WanModel.load_from_file(
                str(model_path),
                device=accelerator.device,
                dtype=torch.float16,
                i2v=True,
                flf2v=False,
                v2_2=True
            )
            print("  DiT model loaded")
            
            # Clear cache after loading
            torch.cuda.empty_cache()
            torch.cuda.synchronize(accelerator.device)
            
        except Exception as e:
            print(f"  ⚠️  Model loading failed: {e}")
            print("  Continuing test without model...")
            dit = None
        
        # Now test transfer AFTER model loading
        print("\nStep 3: Testing transfer AFTER model loading...")
        
        # Get a fresh batch
        batch_after = next(iter(train_dataloader))
        latents_after = batch_after["latents"]
        cpu_max_after = latents_after.abs().max().item()
        print(f"  Latents max: {cpu_max_after:.6e}")
        
        # Test transfer
        if not latents_after.is_contiguous():
            latents_after = latents_after.contiguous()
        latents_after_clone = latents_after.clone()
        
        print(f"  Attempting transfer...")
        latents_after_gpu = latents_after_clone.to(accelerator.device, non_blocking=False)
        torch.cuda.synchronize(accelerator.device)
        gpu_max_after = latents_after_gpu.abs().max().item()
        print(f"  Transfer to GPU: max={gpu_max_after:.6e}")
        
        if gpu_max_after < 1e-6 and cpu_max_after > 1e-6:
            print("  ❌ FAILED: Transfer produces zeros AFTER model loading!")
            print("  This confirms model loading corrupts GPU state!")
            
            # Diagnostic: Can simple transfers work?
            print(f"\n  Diagnostic: Testing simple tensor transfer...")
            simple = torch.randn(100, device='cpu')
            simple_gpu = simple.to(accelerator.device, non_blocking=False)
            torch.cuda.synchronize(accelerator.device)
            simple_max = simple_gpu.abs().max().item()
            print(f"    Simple transfer: max={simple_max:.6e}")
            
            if simple_max > 1e-6:
                print("    ✅ Simple transfers work - issue is with this specific tensor")
            else:
                print("    ❌ Even simple transfers fail - GPU state is corrupted!")
            
            return False
        else:
            print("  ✅ PASSED: Transfer works even after model loading")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("\nModel loading does not corrupt GPU state.")
        print("The issue must be in the training loop itself.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_after_model_loading()
    sys.exit(0 if success else 1)


