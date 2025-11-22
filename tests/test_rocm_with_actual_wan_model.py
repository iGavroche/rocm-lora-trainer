#!/usr/bin/env python3
"""
Test if loading actual WAN model corrupts GPU state.

This test loads the actual WAN model and prepares it with Accelerate,
then checks if tensor transfers still work.
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_with_actual_wan_model():
    """Test with actual WAN model"""
    print("=" * 80)
    print("ROCm Actual WAN Model Test")
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
        
        # Test 1: Transfer before model loading
        print("Test 1: Transfer before WAN model loading")
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
            print("  [FAILED] Transfer fails even before model!")
            return False
        print("  [PASSED]")
        
        # Test 2: Load actual WAN model
        print("\nTest 2: Loading actual WAN model...")
        model_path = Path("models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors")
        if not model_path.exists():
            print(f"  [SKIP] Model not found at {model_path}")
            print("  Cannot test with actual WAN model")
            return True
        
        try:
            from musubi_tuner.hunyuan_model.models import load_transformer
            print("  Loading transformer...")
            # Use same parameters as training code
            attn_mode = "xformers"  # Default from training
            split_attn = False  # Default from training
            loading_device = accelerator.device
            dit_weight_dtype = torch.float16  # Default from training
            dit_in_channels = 16  # WAN default
            transformer = load_transformer(
                str(model_path), 
                attn_mode, 
                split_attn, 
                loading_device, 
                dit_weight_dtype, 
                dit_in_channels
            )
            print(f"  Model loaded: {type(transformer)}")
            
            # Prepare with Accelerate (like training does)
            print("  Preparing with Accelerate...")
            transformer = accelerator.prepare(transformer)
            print("  Model prepared with Accelerate")
            
            allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
            reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
            print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 3: Transfer after model loaded and prepared
        print("\nTest 3: Transfer after WAN model loaded and prepared")
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
            print("  [FAILED] Transfer fails after WAN model loaded!")
            print("  THIS IS THE ROOT CAUSE: WAN model + Accelerate prepare() corrupts GPU state")
            return False
        print("  [PASSED]")
        
        # Test 4: Multiple transfers after model
        print("\nTest 4: Multiple transfers after model (checking for state corruption)")
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
                print(f"  [FAILED] Transfer {i+1} failed - GPU max: {gpu_max:.6e}, CPU max: {cpu_max:.6e}")
                return False
        
        print("  [PASSED] All 3 transfers succeeded")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("[PASS] WAN model loading does not cause transfer failures")
        print("The issue must be something else in the training sequence")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_actual_wan_model()
    sys.exit(0 if success else 1)

