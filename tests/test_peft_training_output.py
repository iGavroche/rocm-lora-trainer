#!/usr/bin/env python3
"""
Test PEFT training to identify important vs verbose output.

This runs a minimal training setup to see what output we get
and identify any issues or important messages.
"""
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reduce verbosity for testing
os.environ["AMD_LOG_LEVEL"] = "2"  # Only warnings and errors
os.environ["ROCM_DEBUG"] = "0"
os.environ["HIP_PROFILE"] = "0"
os.environ["TORCH_LOGS"] = ""
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_DEBUG"] = "0"

def test_peft_training_setup():
    """Test PEFT training setup and identify important output"""
    print("=" * 80)
    print("PEFT Training Output Test")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA/ROCm not available")
        return False
    
    try:
        from accelerate import Accelerator
        from musubi_tuner.dataset.config_utils import (
            load_user_config, BlueprintGenerator, ConfigSanitizer,
            generate_dataset_group_by_blueprint
        )
        from musubi_tuner.networks.peft_lora import create_arch_network
        import argparse
        from multiprocessing import Value
        
        print("[INFO] Testing PEFT training setup...")
        print()
        
        # Initialize Accelerate
        accelerator = Accelerator()
        print(f"[OK] Accelerator initialized: device={accelerator.device}")
        print()
        
        # Load dataset config
        config_path = Path("dataset.toml")
        if not config_path.exists():
            print(f"[SKIP] Dataset config not found at {config_path}")
            print("       This is OK for testing - will skip dataset loading")
            return True
        
        print("[INFO] Loading dataset config...")
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        user_config = load_user_config(str(config_path))
        print("[OK] Dataset config loaded")
        print()
        
        # Create dataset
        print("[INFO] Creating dataset...")
        args = argparse.Namespace()
        blueprint = blueprint_generator.generate(user_config, args, architecture="wan")
        shared_epoch = Value('i', 0)
        train_dataset_group = generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=True,
            num_timestep_buckets=None,
            shared_epoch=shared_epoch
        )
        print(f"[OK] Dataset created: {len(train_dataset_group)} items")
        print()
        
        # Test PEFT network creation
        print("[INFO] Testing PEFT network creation...")
        print("       (This simulates what happens during training)")
        print()
        
        # Create a simple model for testing
        import torch.nn as nn
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(10, 20),
                        nn.ReLU(),
                        nn.Linear(20, 10)
                    ) for _ in range(2)
                ])
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x
        
        model = SimpleTransformer()
        
        # Create PEFT network
        network = create_arch_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=4,
            vae=None,
            text_encoders=None,
            unet=model,
            neuron_dropout=0.1,
        )
        print(f"[OK] PEFT network created: {network.get_nb_trainable_parameters()} trainable params")
        print()
        
        # Test forward pass
        print("[INFO] Testing forward pass...")
        model = model.to(accelerator.device)
        x = torch.randn(2, 10).to(accelerator.device)
        with torch.no_grad():
            y = network.peft_model(x)
        y_max = y.abs().max().item()
        print(f"[OK] Forward pass: output max={y_max:.6e}")
        if y_max < 1e-6:
            print("[ERROR] Output is zeros!")
            return False
        print()
        
        # Test tensor transfer (the critical part)
        print("[INFO] Testing tensor transfer (critical for ROCm bug check)...")
        x_cpu = torch.randn(2, 10)
        cpu_max = x_cpu.abs().max().item()
        x_gpu = x_cpu.to(accelerator.device, non_blocking=False)
        gpu_max = x_gpu.abs().max().item()
        print(f"[OK] Transfer: CPU max={cpu_max:.6e}, GPU max={gpu_max:.6e}")
        if gpu_max < 1e-6 and cpu_max > 1e-6:
            print("[ERROR] Transfer produced zeros (ROCm bug detected!)")
            return False
        print()
        
        print("=" * 80)
        print("[SUCCESS] All PEFT tests passed!")
        print("=" * 80)
        print()
        print("IMPORTANT OUTPUT TO LOOK FOR IN TRAINING:")
        print("  [OK] = Everything working correctly")
        print("  [ERROR] = Critical issue - training will fail")
        print("  [WARN] = Warning - may indicate issues but training might continue")
        print()
        print("VERBOSE OUTPUT (can be ignored):")
        print("  - ROCm HIP logging (comgrctx, hip_context, etc.)")
        print("  - PyTorch TorchDynamo metrics")
        print("  - Triton warnings about cuobjdump/nvdisasm")
        print("  - TorchDynamo compilation metrics")
        print()
        print("CRITICAL CHECKS DURING TRAINING:")
        print("  1. Look for 'PEFT LoRA applied' message")
        print("  2. Check that tensor transfers don't produce zeros")
        print("  3. Verify loss is not zero")
        print("  4. Check for any [ERROR] messages")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_peft_training_setup()
    sys.exit(0 if success else 1)




