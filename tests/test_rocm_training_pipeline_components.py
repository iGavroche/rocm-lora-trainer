#!/usr/bin/env python3
"""
Systematic component testing for ROCm training pipeline issues.

Tests each component that could cause tensors to become zeros:
1. Cache file loading
2. CPU to GPU transfer (isolated)
3. DataLoader behavior
4. Accelerate interaction
5. Tensor stacking
6. Batch operations
7. Memory pressure scenarios

Run with: python tests/test_rocm_training_pipeline_components.py
"""
import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_component_1_cache_loading():
    """Test 1: Cache file loading to CPU"""
    print("=" * 80)
    print("COMPONENT 1: Cache File Loading to CPU")
    print("=" * 80)
    
    try:
        from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
        
        # Find a cache file
        cache_dir = Path("myface")
        cache_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
        
        if not cache_files:
            print("  ⚠️  SKIPPED: No cache files found in myface/")
            return True
        
        cache_file = cache_files[0]
        print(f"  Testing with: {cache_file}")
        
        with MemoryEfficientSafeOpen(str(cache_file), disable_numpy_memmap=False) as f:
            keys = list(f.keys())
            print(f"  Keys found: {keys}")
            
            for key in keys[:2]:  # Test first 2 keys
                tensor = f.get_tensor(key, device=torch.device("cpu"), dtype=torch.float32)
                max_val = tensor.abs().max().item()
                mean_val = tensor.abs().mean().item()
                
                print(f"  {key}:")
                print(f"    Shape: {tensor.shape}, dtype: {tensor.dtype}")
                print(f"    Max: {max_val:.6e}, Mean: {mean_val:.6e}")
                
                if max_val < 1e-6:
                    print(f"    ❌ FAILED: Tensor is all zeros after loading!")
                    return False
                else:
                    print(f"    ✅ PASSED: Tensor has valid values")
        
        print("  ✅ COMPONENT 1 PASSED: Cache loading works correctly")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_2_cpu_to_gpu_isolated():
    """Test 2: CPU to GPU transfer in isolation"""
    print("\n" + "=" * 80)
    print("COMPONENT 2: CPU to GPU Transfer (Isolated)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA not available")
        return True
    
    device = torch.device("cuda:0")
    
    try:
        # Test with same shape as training tensors
        shape = (1, 16, 1, 64, 64)
        print(f"  Testing with shape: {shape}")
        
        # Create on CPU
        cpu_tensor = torch.randn(shape, device='cpu')
        cpu_max = cpu_tensor.abs().max().item()
        print(f"  CPU tensor max: {cpu_max:.6e}")
        
        # Transfer to GPU using different methods
        methods = [
            ("Direct .to()", lambda t: t.to(device)),
            ("Pinned memory + .to()", lambda t: t.pin_memory().to(device, non_blocking=True)),
            ("copy_() method", lambda t: torch.empty(t.shape, dtype=t.dtype, device=device).copy_(t)),
        ]
        
        for method_name, method_func in methods:
            try:
                gpu_tensor = method_func(cpu_tensor)
                torch.cuda.synchronize(device)
                gpu_max = gpu_tensor.abs().max().item()
                
                print(f"  {method_name}:")
                print(f"    GPU max: {gpu_max:.6e}")
                
                if gpu_max < 1e-6 and cpu_max > 1e-6:
                    print(f"    ❌ FAILED: Tensor became zeros!")
                    return False
                elif abs(gpu_max - cpu_max) > 1e-3:
                    print(f"    ⚠️  WARNING: Values changed (CPU={cpu_max:.6e}, GPU={gpu_max:.6e})")
                else:
                    print(f"    ✅ PASSED: Transfer successful")
            except Exception as e:
                print(f"    ❌ ERROR with {method_name}: {e}")
                return False
        
        print("  ✅ COMPONENT 2 PASSED: CPU to GPU transfer works in isolation")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_3_dataloader_behavior():
    """Test 3: DataLoader behavior with cache files"""
    print("\n" + "=" * 80)
    print("COMPONENT 3: DataLoader Behavior")
    print("=" * 80)
    
    try:
        from musubi_tuner.dataset.image_video_dataset import ImageDataset
        from musubi_tuner.dataset.config_utils import load_user_config, BlueprintGenerator, ConfigSanitizer
        import argparse
        
        # Load dataset config
        config_path = Path("dataset.toml")
        if not config_path.exists():
            print("  ⚠️  SKIPPED: dataset.toml not found")
            return True
        
        # Create a minimal argparse.Namespace for blueprint generation
        args = argparse.Namespace()
        
        from musubi_tuner.dataset.config_utils import generate_dataset_group_by_blueprint
        
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        user_config = load_user_config(str(config_path))
        blueprint = blueprint_generator.generate(user_config, args, architecture="wan")
        
        # Create a shared_epoch object for testing
        from multiprocessing import Value
        shared_epoch = Value('i', 0)
        
        # Generate actual dataset instances
        train_dataset_group = generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=True,
            num_timestep_buckets=None,
            shared_epoch=shared_epoch
        )
        
        if len(train_dataset_group.datasets) == 0:
            print("  ⚠️  SKIPPED: No datasets found")
            return True
        
        dataset = train_dataset_group.datasets[0]
        print(f"  Dataset: {type(dataset).__name__}")
        print(f"  Dataset length: {len(dataset)}")
        
        # Test getting a batch directly from dataset
        if len(dataset) > 0:
            batch = dataset[0]
            print(f"  Batch keys: {list(batch.keys())}")
            
            if "latents" in batch:
                latents = batch["latents"]
                max_val = latents.abs().max().item()
                print(f"  Latents from dataset[0]:")
                print(f"    Shape: {latents.shape}, dtype: {latents.dtype}, device: {latents.device}")
                print(f"    Max: {max_val:.6e}")
                
                if max_val < 1e-6:
                    print(f"    ❌ FAILED: Latents are zeros from dataset!")
                    return False
                else:
                    print(f"    ✅ PASSED: Latents have valid values")
            
            # Test DataLoader with num_workers=0 (same as training)
            # Use the same collator as training
            from musubi_tuner.hv_train_network import collator_class
            from multiprocessing import Value
            current_epoch = Value('i', 0)
            collator = collator_class(current_epoch, dataset)
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collator
            )
            
            batch_from_loader = next(iter(dataloader))
            print(f"  Batch from DataLoader:")
            print(f"    Keys: {list(batch_from_loader.keys())}")
            
            if "latents" in batch_from_loader:
                latents_loader = batch_from_loader["latents"]
                max_val_loader = latents_loader.abs().max().item()
                print(f"    Latents:")
                print(f"      Shape: {latents_loader.shape}, dtype: {latents_loader.dtype}, device: {latents_loader.device}")
                print(f"      Max: {max_val_loader:.6e}")
                
                if max_val_loader < 1e-6:
                    print(f"      ❌ FAILED: Latents are zeros from DataLoader!")
                    return False
                else:
                    print(f"      ✅ PASSED: Latents have valid values from DataLoader")
            
            print("  ✅ COMPONENT 3 PASSED: DataLoader produces valid batches")
            return True
        else:
            print("  ⚠️  SKIPPED: Dataset is empty")
            return True
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_4_accelerate_interaction():
    """Test 4: Accelerate interaction with tensors"""
    print("\n" + "=" * 80)
    print("COMPONENT 4: Accelerate Interaction")
    print("=" * 80)
    
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        device = accelerator.device
        print(f"  Accelerator device: {device}")
        
        # Create tensor on CPU
        cpu_tensor = torch.randn((1, 16, 1, 64, 64), device='cpu')
        cpu_max = cpu_tensor.abs().max().item()
        print(f"  CPU tensor max: {cpu_max:.6e}")
        
        # Test 1: Prepare tensor with Accelerate
        try:
            prepared_tensor = accelerator.prepare(cpu_tensor)
            prepared_max = prepared_tensor.abs().max().item()
            print(f"  After accelerator.prepare():")
            print(f"    Device: {prepared_tensor.device}, Max: {prepared_max:.6e}")
            
            if prepared_max < 1e-6 and cpu_max > 1e-6:
                print(f"    ❌ FAILED: Tensor became zeros after prepare()!")
                return False
            else:
                print(f"    ✅ PASSED: Tensor preserved after prepare()")
        except Exception as e:
            print(f"    ⚠️  WARNING: accelerator.prepare() failed: {e}")
        
        # Test 2: Manual move to accelerator device
        manual_tensor = cpu_tensor.to(device)
        manual_max = manual_tensor.abs().max().item()
        print(f"  Manual .to(accelerator.device):")
        print(f"    Max: {manual_max:.6e}")
        
        if manual_max < 1e-6 and cpu_max > 1e-6:
            print(f"    ❌ FAILED: Tensor became zeros after manual move!")
            return False
        else:
            print(f"    ✅ PASSED: Manual move successful")
        
        print("  ✅ COMPONENT 4 PASSED: Accelerate interaction works")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_5_tensor_stacking():
    """Test 5: Tensor stacking operations"""
    print("\n" + "=" * 80)
    print("COMPONENT 5: Tensor Stacking")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA not available")
        return True
    
    device = torch.device("cuda:0")
    
    try:
        # Create multiple tensors (simulating batch)
        tensors = []
        for i in range(3):
            tensor = torch.randn((16, 1, 64, 64), device='cpu')
            max_val = tensor.abs().max().item()
            tensors.append(tensor)
            print(f"  Tensor {i}: CPU max={max_val:.6e}")
        
        # Stack on CPU
        stacked_cpu = torch.stack(tensors)
        stacked_cpu_max = stacked_cpu.abs().max().item()
        print(f"  Stacked on CPU: max={stacked_cpu_max:.6e}")
        
        if stacked_cpu_max < 1e-6:
            print(f"    ❌ FAILED: Stacking on CPU produces zeros!")
            return False
        
        # Move stacked tensor to GPU
        stacked_gpu = stacked_cpu.to(device)
        stacked_gpu_max = stacked_gpu.abs().max().item()
        print(f"  Stacked tensor moved to GPU: max={stacked_gpu_max:.6e}")
        
        if stacked_gpu_max < 1e-6 and stacked_cpu_max > 1e-6:
            print(f"    ❌ FAILED: Stacked tensor became zeros when moved to GPU!")
            return False
        
        # Alternative: Stack on GPU
        tensors_gpu = [t.to(device) for t in tensors]
        stacked_gpu_direct = torch.stack(tensors_gpu)
        stacked_gpu_direct_max = stacked_gpu_direct.abs().max().item()
        print(f"  Stacked directly on GPU: max={stacked_gpu_direct_max:.6e}")
        
        if stacked_gpu_direct_max < 1e-6:
            print(f"    ❌ FAILED: Stacking directly on GPU produces zeros!")
            return False
        
        print("  ✅ COMPONENT 5 PASSED: Tensor stacking works correctly")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_6_memory_pressure():
    """Test 6: Memory pressure scenarios"""
    print("\n" + "=" * 80)
    print("COMPONENT 6: Memory Pressure Scenarios")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA not available")
        return True
    
    device = torch.device("cuda:0")
    
    try:
        shape = (1, 16, 1, 64, 64)
        
        # Allocate multiple tensors to simulate memory pressure
        print("  Allocating multiple tensors to simulate memory pressure...")
        tensors = []
        for i in range(10):
            tensor = torch.randn(shape, device=device)
            max_val = tensor.abs().max().item()
            tensors.append(tensor)
            if i < 3:
                print(f"    Tensor {i}: max={max_val:.6e}")
        
        # Now try CPU to GPU transfer under memory pressure
        cpu_tensor = torch.randn(shape, device='cpu')
        cpu_max = cpu_tensor.abs().max().item()
        print(f"  CPU tensor max: {cpu_max:.6e}")
        
        gpu_tensor = cpu_tensor.to(device)
        torch.cuda.synchronize(device)
        gpu_max = gpu_tensor.abs().max().item()
        print(f"  GPU tensor max (under memory pressure): {gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print(f"    ❌ FAILED: Transfer fails under memory pressure!")
            return False
        
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        
        print("  ✅ COMPONENT 6 PASSED: Memory pressure doesn't cause corruption")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_7_full_pipeline_simulation():
    """Test 7: Full pipeline simulation (cache -> CPU -> GPU -> operation)"""
    print("\n" + "=" * 80)
    print("COMPONENT 7: Full Pipeline Simulation")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA not available")
        return True
    
    try:
        from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
        
        device = torch.device("cuda:0")
        cache_dir = Path("myface")
        cache_files = list(cache_dir.glob("*.safetensors")) if cache_dir.exists() else []
        
        if not cache_files:
            print("  ⚠️  SKIPPED: No cache files found")
            return True
        
        cache_file = cache_files[0]
        print(f"  Simulating full pipeline with: {cache_file}")
        
        # Step 1: Load from cache to CPU
        with MemoryEfficientSafeOpen(str(cache_file), disable_numpy_memmap=False) as f:
            keys = list(f.keys())
            if not keys:
                print("  ⚠️  SKIPPED: No keys in cache file")
                return True
            
            key = keys[0]
            tensor_cpu = f.get_tensor(key, device=torch.device("cpu"), dtype=torch.float32)
            cpu_max = tensor_cpu.abs().max().item()
            print(f"  Step 1 - Loaded to CPU: max={cpu_max:.6e}")
            
            if cpu_max < 1e-6:
                print(f"    ❌ FAILED: Cache file has zeros!")
                return False
        
        # Step 2: Move to GPU
        tensor_gpu = tensor_cpu.to(device)
        torch.cuda.synchronize(device)
        gpu_max = tensor_gpu.abs().max().item()
        print(f"  Step 2 - Moved to GPU: max={gpu_max:.6e}")
        
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print(f"    ❌ FAILED: Transfer to GPU produces zeros!")
            return False
        
        # Step 3: Perform operation (like torch.randn_like)
        noise = torch.randn_like(tensor_gpu)
        noise_max = noise.abs().max().item()
        print(f"  Step 3 - torch.randn_like: max={noise_max:.6e}")
        
        if noise_max < 1e-6:
            print(f"    ❌ FAILED: torch.randn_like produces zeros!")
            return False
        
        print("  ✅ COMPONENT 7 PASSED: Full pipeline works correctly")
        return True
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all component tests"""
    print("\n" + "=" * 80)
    print("ROCm Training Pipeline Component Tests")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        if hasattr(torch.version, 'hip'):
            print(f"ROCm: {torch.version.hip}")
    print()
    
    results = {}
    results[1] = test_component_1_cache_loading()
    results[2] = test_component_2_cpu_to_gpu_isolated()
    results[3] = test_component_3_dataloader_behavior()
    results[4] = test_component_4_accelerate_interaction()
    results[5] = test_component_5_tensor_stacking()
    results[6] = test_component_6_memory_pressure()
    results[7] = test_component_7_full_pipeline_simulation()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for i, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  Component {i}: {status}")
    
    print(f"\nTotal: {passed}/{total} components passed")
    
    if passed == total:
        print("\n✅ ALL COMPONENTS PASSED")
        print("If training still fails, the issue is likely:")
        print("  1. Specific sequence of operations in training loop")
        print("  2. Interaction between multiple components")
        print("  3. State-dependent behavior (only fails after certain operations)")
        return True
    else:
        print("\n❌ SOME COMPONENTS FAILED")
        print("The failing components indicate where the problem lies.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

