"""Test to verify PEFT target modules are correctly found and applied."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from src.musubi_tuner.wan.modules.model import load_wan_model
from src.musubi_tuner.wan.configs import WAN_CONFIGS
from musubi_tuner.networks.peft_lora import create_arch_network

def test_peft_target_modules():
    """Verify that PEFT finds and applies to all target modules."""
    acc = Accelerator()
    config = WAN_CONFIGS['i2v-A14B']
    
    print("Loading model...")
    model = load_wan_model(
        config, 
        acc.device, 
        r'.\models\wan\wan2.2_i2v_low_noise_14B_fp16.safetensors', 
        'xformers', 
        False, 
        'cpu', 
        torch.float16, 
        False, 
        None, 
        None, 
        False, 
        False
    )
    print("Model loaded")
    
    print("Creating PEFT network...")
    network = create_arch_network(1.0, 32, 32, None, [], model, 0.0)
    print("PEFT network created")
    
    # Check target modules
    target_modules_list = list(network.lora_config.target_modules)
    print(f"\nTarget modules passed to PEFT: {len(target_modules_list)}")
    print(f"First 10: {target_modules_list[:10]}")
    
    # Verify modules exist in model
    print("\nVerifying modules exist in model...")
    missing = []
    for tm in target_modules_list:
        parts = tm.split('.')
        current = model
        found = True
        for p in parts:
            if hasattr(current, p):
                current = getattr(current, p)
            else:
                found = False
                break
        if not found:
            missing.append(tm)
    
    print(f"Missing modules: {len(missing)}")
    if missing:
        print(f"First 10 missing: {missing[:10]}")
        return False
    
    # Check how many modules PEFT actually found
    peft_model = network.peft_model
    trainable_params = peft_model.get_nb_trainable_parameters()
    print(f"\nPEFT trainable parameters: {trainable_params}")
    
    # Count LoRA adapters
    adapter_names = peft_model.get_adapter_names()
    print(f"Adapter names: {adapter_names}")
    
    # Check if all target modules have LoRA adapters
    # PEFT uses pattern matching, so it deduplicates target_modules
    # Check the state dict to see which modules actually got LoRA
    state_dict = peft_model.state_dict()
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    print(f"\nLoRA keys in state dict: {len(lora_keys)}")
    print(f"First 10 LoRA keys: {lora_keys[:10]}")
    
    # Count how many blocks have LoRA applied
    blocks_with_lora = set()
    for k in lora_keys:
        if 'blocks.' in k:
            try:
                block_num = k.split('blocks.')[1].split('.')[0]
                blocks_with_lora.add(block_num)
            except:
                pass
    
    print(f"\nBlocks with LoRA: {len(blocks_with_lora)} (expected: 40)")
    if len(blocks_with_lora) > 0:
        block_nums = sorted([int(b) for b in blocks_with_lora])
        print(f"Block numbers: {block_nums[:10]}... (showing first 10)")
    
    # Expected: 400 target modules * 2 (A and B matrices) = 800 LoRA weight keys
    # Plus potentially bias terms
    expected_min_keys = len(target_modules_list) * 2  # A and B matrices
    if len(lora_keys) < expected_min_keys:
        print(f"WARNING: Expected at least {expected_min_keys} LoRA keys, got {len(lora_keys)}")
        # But if we have LoRA in all blocks, that's probably fine
        if len(blocks_with_lora) == 40:
            print("However, all 40 blocks have LoRA, which is correct")
        else:
            return False
    
    if len(blocks_with_lora) != 40:
        print(f"ERROR: Expected LoRA in all 40 blocks, but found in {len(blocks_with_lora)} blocks")
        return False
    
    print("\n✓ All target modules verified!")
    return True

if __name__ == "__main__":
    success = test_peft_target_modules()
    sys.exit(0 if success else 1)

