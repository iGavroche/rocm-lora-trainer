#!/usr/bin/env python3
"""Test script to verify safetensors loading works with safetensors 0.7.0+"""
import sys
import os
import time
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_safetensors

def test_model_loading():
    """Test loading the WAN i2v model"""
    model_path = "models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the model file exists before testing.")
        return False
    
    print(f"Testing safetensors loading with file: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024**3):.2f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Test 1: Load using load_safetensors (for T5/VAE)
    print("=" * 60)
    print("TEST 1: Loading with load_safetensors (T5/VAE style)")
    print("=" * 60)
    start_time = time.time()
    try:
        state_dict = load_safetensors(model_path, device="cpu", disable_mmap=False)
        load_time = time.time() - start_time
        print(f"✓ Successfully loaded {len(state_dict)} tensors to CPU in {load_time:.2f}s")
        
        # Test GPU transfer
        if torch.cuda.is_available():
            print(f"\nTransferring to {device}...")
            transfer_start = time.time()
            for i, (key, value) in enumerate(state_dict.items()):
                if i % 100 == 0:
                    print(f"  Transferred {i}/{len(state_dict)} tensors...")
                state_dict[key] = value.to(device, non_blocking=True)
            torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            print(f"✓ Successfully transferred to GPU in {transfer_time:.2f}s")
        
        print(f"Total time: {time.time() - start_time:.2f}s")
        print()
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Load using load_safetensors_with_lora_and_fp8 (for DiT model)
    print("=" * 60)
    print("TEST 2: Loading with load_safetensors_with_lora_and_fp8 (DiT style)")
    print("=" * 60)
    start_time = time.time()
    state_dict2 = None
    try:
        state_dict2 = load_safetensors_with_lora_and_fp8(
            model_files=model_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=False,
            calc_device=device,
            move_to_device=True,
            dit_weight_dtype=None,
            disable_numpy_memmap=False,
        )
        load_time = time.time() - start_time
        print(f"✓ Successfully loaded {len(state_dict2)} tensors in {load_time:.2f}s")
        
        # Verify tensors are on correct device
        sample_key = list(state_dict2.keys())[0]
        sample_device = state_dict2[sample_key].device
        # Normalize device comparison (cuda:0 == cuda)
        expected_device_str = str(device)
        sample_device_str = str(sample_device)
        # Remove :0 suffix for comparison as cuda:0 == cuda
        expected_normalized = expected_device_str.replace(':0', '')
        sample_normalized = sample_device_str.replace(':0', '')
        
        print(f"Sample tensor device: {sample_device} (expected: {device})")
        
        if sample_normalized == expected_normalized or sample_device == device:
            print("✓ All tensors on correct device")
        else:
            print(f"✗ WARNING: Tensors on wrong device! ({sample_device} != {device})")
        
        # Clean up
        del state_dict2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print()
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        if state_dict2 is not None:
            del state_dict2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        return False
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)


