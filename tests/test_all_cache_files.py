#!/usr/bin/env python3
"""
Test ALL cache files to find which ones are corrupted.

This will identify exactly which cache files have zeros.
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_all_cache_files():
    """Test all cache files in the dataset"""
    print("=" * 80)
    print("Testing ALL Cache Files")
    print("=" * 80)
    
    try:
        from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
        
        # Find all cache files
        cache_dir = Path("myface")
        if not cache_dir.exists():
            print("ERROR: myface directory not found")
            return False
        
        cache_files = list(cache_dir.glob("*.safetensors"))
        print(f"Found {len(cache_files)} cache files")
        print()
        
        corrupted_files = []
        valid_files = []
        
        for cache_file in cache_files:
            print(f"Testing: {cache_file.name}")
            try:
                with MemoryEfficientSafeOpen(str(cache_file), disable_numpy_memmap=False) as f:
                    keys = list(f.keys())
                    
                    all_zeros = True
                    for key in keys:
                        tensor = f.get_tensor(key, device=torch.device("cpu"), dtype=torch.float32)
                        max_val = tensor.abs().max().item()
                        
                        if max_val < 1e-6:
                            print(f"  ❌ {key}: ALL ZEROS (max={max_val:.6e})")
                        else:
                            print(f"  ✅ {key}: max={max_val:.6e}")
                            all_zeros = False
                    
                    if all_zeros:
                        corrupted_files.append(cache_file)
                        print(f"  ❌ FILE CORRUPTED: All tensors are zeros")
                    else:
                        valid_files.append(cache_file)
                        print(f"  ✅ FILE VALID")
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                corrupted_files.append(cache_file)
            
            print()
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files: {len(cache_files)}")
        print(f"Valid files: {len(valid_files)}")
        print(f"Corrupted files: {len(corrupted_files)}")
        
        if corrupted_files:
            print(f"\n❌ CORRUPTED FILES FOUND:")
            for f in corrupted_files:
                print(f"  - {f.name}")
            print(f"\nACTION REQUIRED: Delete these files and re-encode:")
            print(f"  python src/musubi_tuner/wan_cache_latents.py --dataset_config dataset.toml --vae models/wan/wan_2.1_vae.safetensors --i2v")
            return False
        else:
            print(f"\n✅ ALL CACHE FILES ARE VALID")
            return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_cache_files()
    sys.exit(0 if success else 1)


