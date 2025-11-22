#!/usr/bin/env python3
"""
Test tensor transfer with model loaded on GPU.

This tests if loading the model corrupts GPU state or causes tensor transfer failures.
"""
import torch
import sys
from pathlib import Path
from multiprocessing import Value

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_with_model_loaded():
    """Test if model loading causes tensor corruption"""
    print("=" * 80)
    print("ROCm Tensor Transfer with Model Loaded Test")
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
        print("Test 1: Transfer before model loading")
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
        
        # Test 2: Simulate model loading (allocate large tensor)
        print("\nTest 2: Simulating model loading (allocating large tensor)")
        model_sim = torch.randn((10000, 10000), device=accelerator.device)
        allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
        print(f"  Allocated {allocated:.2f} GB to simulate model")
        
        # Test 3: Transfer after model "loaded"
        print("\nTest 3: Transfer after model loaded")
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
            print("  [FAILED] Transfer fails after model loaded!")
            print("  This indicates model loading corrupts GPU state")
            return False
        print("  [PASSED]")
        
        # Test 4: Multiple transfers after model
        print("\nTest 4: Multiple transfers after model (checking for memory fragmentation)")
        for i in range(5):
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
        
        print("  [PASSED] All 5 transfers succeeded")
        
        # Clean up
        del model_sim
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("[PASS] All tests passed - model loading does not cause transfer failures")
        print("The issue must occur during actual model forward/backward pass")
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_model_loaded()
    sys.exit(0 if success else 1)

