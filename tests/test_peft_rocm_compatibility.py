#!/usr/bin/env python3
"""
Test PEFT (Hugging Face Parameter-Efficient Fine-Tuning) compatibility on Windows + ROCm.

This test verifies:
1. PEFT can be installed and imported
2. Basic LoRA operations work on ROCm
3. Tensor transfers don't produce zeros (the known ROCm bug)
4. LoRA training loop works correctly
"""
import torch
import sys
from pathlib import Path

def test_peft_installation():
    """Test 1: Verify PEFT can be installed and imported"""
    print("=" * 80)
    print("Test 1: PEFT Installation and Import")
    print("=" * 80)
    
    try:
        import peft
        from peft import LoraConfig, get_peft_model
        print(f"[PASS] PEFT version: {peft.__version__}")
        print("[PASS] PEFT imported successfully")
        return True, peft
    except ImportError as e:
        print(f"[FAIL] PEFT not installed: {e}")
        print("   Install with: pip install peft")
        return False, None

def test_peft_basic_operations():
    """Test 2: Basic PEFT operations on ROCm"""
    print("\n" + "=" * 80)
    print("Test 2: Basic PEFT Operations on ROCm")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    
    try:
        from peft import LoraConfig, get_peft_model
        import torch.nn as nn
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 10)
            
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x
        
        model = SimpleModel()
        print(f"[PASS] Created test model: {model}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear1", "linear2"],
            lora_dropout=0.1,
        )
        print(f"[PASS] Created LoRA config: {lora_config}")
        
        # Apply LoRA to model
        peft_model = get_peft_model(model, lora_config)
        print(f"[PASS] Applied LoRA to model")
        print(f"   Trainable parameters: {peft_model.get_nb_trainable_parameters()}")
        
        # Move model to GPU
        peft_model = peft_model.to(device)
        print(f"[PASS] Moved model to {device}")
        
        # Test forward pass
        x = torch.randn(2, 10).to(device)
        print(f"[PASS] Created input tensor on {device}: shape={x.shape}, max={x.abs().max().item():.6e}")
        
        with torch.no_grad():
            y = peft_model(x)
        
        y_max = y.abs().max().item()
        print(f"[PASS] Forward pass completed: output max={y_max:.6e}")
        
        if y_max < 1e-6:
            print("[FAIL] Output is zeros - this indicates the ROCm bug!")
            return False
        
        print("[PASS] Basic PEFT operations work correctly on ROCm")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error in basic operations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft_tensor_transfers():
    """Test 3: Tensor transfers with PEFT (check for ROCm bug)"""
    print("\n" + "=" * 80)
    print("Test 3: Tensor Transfers with PEFT (ROCm Bug Check)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    
    try:
        from peft import LoraConfig, get_peft_model
        import torch.nn as nn
        
        # Create model with LoRA
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        lora_config = LoraConfig(r=4, target_modules=["linear"])
        peft_model = get_peft_model(model, lora_config).to(device)
        
        # Test multiple CPU→GPU transfers
        print("Testing multiple CPU->GPU transfers...")
        all_passed = True
        
        for i in range(5):
            # Create tensor on CPU
            x_cpu = torch.randn(1, 10)
            cpu_max = x_cpu.abs().max().item()
            
            # Transfer to GPU
            x_gpu = x_cpu.to(device)
            gpu_max = x_gpu.abs().max().item()
            
            print(f"  Transfer {i+1}: CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e}")
            
            if gpu_max < 1e-6 and cpu_max > 1e-6:
                print(f"  [FAIL] Transfer {i+1} failed - zeros detected (ROCm bug)")
                all_passed = False
                break
        
        if all_passed:
            print("[PASS] All tensor transfers successful - no ROCm bug detected")
            return True
        else:
            print("[FAIL] Tensor transfers failed - ROCm bug confirmed")
            return False
        
    except Exception as e:
        print(f"❌ Error in tensor transfer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft_training_loop():
    """Test 4: Simple training loop with PEFT"""
    print("\n" + "=" * 80)
    print("Test 4: Simple Training Loop with PEFT")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/ROCm not available")
        return False
    
    device = torch.device("cuda:0")
    
    try:
        from peft import LoraConfig, get_peft_model
        import torch.nn as nn
        import torch.optim as optim
        
        # Create model with LoRA
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        lora_config = LoraConfig(r=4, target_modules=["linear"])
        peft_model = get_peft_model(model, lora_config).to(device)
        
        optimizer = optim.AdamW(peft_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        print("Running 3 training steps...")
        
        for step in range(3):
            # Create batch on CPU
            x_cpu = torch.randn(2, 10)
            target_cpu = torch.randn(2, 10)
            
            # Transfer to GPU
            x_gpu = x_cpu.to(device)
            target_gpu = target_cpu.to(device)
            
            # Check for zeros
            if x_gpu.abs().max().item() < 1e-6:
                print(f"  [FAIL] Step {step+1}: Input became zeros (ROCm bug)")
                return False
            
            # Forward pass
            output = peft_model(x_gpu)
            
            if output.abs().max().item() < 1e-6:
                print(f"  [FAIL] Step {step+1}: Output is zeros")
                return False
            
            # Loss
            loss = criterion(output, target_gpu)
            
            if loss.item() < 1e-6:
                print(f"  [WARN] Step {step+1}: Loss is very small ({loss.item():.6e})")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"  [PASS] Step {step+1}: Loss={loss.item():.6e}, Output max={output.abs().max().item():.6e}")
        
        print("[PASS] Training loop completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in training loop: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all PEFT compatibility tests"""
    print("=" * 80)
    print("PEFT ROCm Compatibility Test Suite")
    print("=" * 80)
    print()
    
    results = {}
    
    # Test 1: Installation
    success, peft_module = test_peft_installation()
    results["installation"] = success
    if not success:
        print("\n[FAIL] PEFT installation failed. Please install PEFT first:")
        print("   pip install peft")
        return False
    
    # Test 2: Basic operations
    results["basic_operations"] = test_peft_basic_operations()
    
    # Test 3: Tensor transfers
    results["tensor_transfers"] = test_peft_tensor_transfers()
    
    # Test 4: Training loop
    results["training_loop"] = test_peft_training_loop()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[PASS] ALL TESTS PASSED - PEFT is compatible with Windows + ROCm")
        print("   Proceed with PEFT migration")
    else:
        print("[FAIL] SOME TESTS FAILED - PEFT may have issues on Windows + ROCm")
        print("   Consider testing Torchtune or alternative solutions")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

