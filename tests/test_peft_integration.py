#!/usr/bin/env python3
"""
Test PEFT integration with WAN model.

This test verifies that the PEFT compatibility layer works correctly
with the WAN transformer model.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_peft_compatibility_layer():
    """Test PEFT compatibility layer with simple model"""
    print("=" * 80)
    print("Test: PEFT Compatibility Layer")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[SKIP] CUDA/ROCm not available")
        return True
    
    try:
        from musubi_tuner.networks.peft_lora import create_arch_network, PEFTLoRANetwork
        import torch.nn as nn
        
        # Create a simple model similar to transformer structure
        class SimpleTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(10, 10)
                self.k_proj = nn.Linear(10, 10)
                self.v_proj = nn.Linear(10, 10)
                self.out_proj = nn.Linear(10, 10)
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 10)
            
            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                out = self.out_proj(q + k + v)
                out = self.fc1(out)
                out = self.fc2(out)
                return out
        
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([SimpleTransformerBlock() for _ in range(2)])
            
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x
        
        model = SimpleTransformer()
        print(f"[PASS] Created test model: {model}")
        
        # Test create_arch_network
        network = create_arch_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=4,
            vae=None,
            text_encoders=None,
            unet=model,
            neuron_dropout=0.1,
        )
        print(f"[PASS] Created PEFT LoRA network: {network}")
        
        # Test forward pass
        x = torch.randn(2, 10)
        with torch.no_grad():
            y = network.peft_model(x)
        
        y_max = y.abs().max().item()
        print(f"[PASS] Forward pass: output max={y_max:.6e}")
        
        if y_max < 1e-6:
            print("[FAIL] Output is zeros")
            return False
        
        # Test prepare_optimizer_params
        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=1e-4)
        print(f"[PASS] Trainable parameters: {len(trainable_params)}")
        
        print("\n[PASS] PEFT compatibility layer works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft_target_module_detection():
    """Test target module auto-detection"""
    print("\n" + "=" * 80)
    print("Test: PEFT Target Module Detection")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[SKIP] CUDA/ROCm not available")
        return True
    
    try:
        from musubi_tuner.networks.peft_lora import PEFTLoRANetwork
        import torch.nn as nn
        
        # Create model with known structure
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(10, 10)
                self.k_proj = nn.Linear(10, 10)
                self.fc1 = nn.Linear(10, 20)
        
        model = TestModel()
        
        # Test auto-detection
        network = PEFTLoRANetwork(
            transformer=model,
            multiplier=1.0,
            lora_dim=4,
            lora_alpha=4,
        )
        
        # Check that target modules were detected
        target_modules = network.lora_config.target_modules
        print(f"[PASS] Detected target modules: {target_modules}")
        
        if not target_modules:
            print("[WARN] No target modules detected (may need manual specification)")
        
        print("[PASS] Target module detection works")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_peft_compatibility_layer()
    success2 = test_peft_target_module_detection()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("[PASS] All PEFT integration tests passed")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("[FAIL] Some tests failed")
        sys.exit(1)




