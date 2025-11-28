#!/bin/bash
# Train Z-Image LoRA with SAFE settings to prevent driver crashes
# Using Z-Image-Turbo model
#
# IMPORTANT: Before training, you must create caches:
#   1. Run ./z_image_cache_latents.sh to cache VAE latents
#   2. Run ./z_image_cache_text_encoder_outputs.sh to cache text encoder outputs
#   3. Then run this script to start training

# Activate virtual environment
source .venv/bin/activate

# ROCm stability: Disable XNACK for RDNA3.5 (Strix Halo)
# XNACK can cause instability with ROCm on RDNA3.5
export HSA_XNACK=0

export HSA_ENABLE_SDMA=0  # Disables SDMA, fixes allocation faults in training loops.
export PYTORCH_ROCM_ARCH=gfx1151  # Replace with your GPU arch (e.g., gfx1030 for RX 6000; check `rocm-smi`).
# NOTE: torch.compile is DISABLED for gfx1151 (Strix Halo) due to known system freeze issue
# See: https://github.com/ROCm/TheRock/issues/1937
# Do NOT use --compile flag on gfx1151 - it causes system freeze on script exit
# export TORCH_LOGS="+dynamo"  # Only enable if using --compile (NOT recommended on gfx1151)
# export TORCH_COMPILE_DEBUG=1  # Only enable if using --compile (NOT recommended on gfx1151)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# Additional ROCm stability settings for Strix Halo
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # Enable experimental ROCm features
export HIP_FORCE_DEV_KERNELS=1  # Force device kernels (may help with SIGSEGV)
export PYTORCH_ENABLE_MPS_FALLBACK=0  # Disable MPS fallback

# New ROCm stability variables (November 2025 fixes for Strix Halo SIGSEGV)
# Disable SDMA completely (more aggressive than HSA_ENABLE_SDMA=0)
export HSA_AMD_SDMA_ENABLE=0
# Force single device to avoid multi-device issues
export HIP_VISIBLE_DEVICES=0
# Use stream-ordered allocator (new in ROCm 7.1+) - may help with List[Tensor] operations
export PYTORCH_ROCM_FORCE_STREAM_ORDERED_ALLOCATOR=1

# PyTorch memory management - helps with fragmentation on UMA systems
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
DATASET_CONFIG="dataset.toml"  # Using full dataset config
OUTPUT_DIR="outputs"
OUTPUT_NAME="z_image_lora"

# Model paths - can use local directory or HuggingFace model ID
MODEL_PATH="models/z-image"  # Local model directory, or use "Tongyi-MAI/Z-Image-Turbo" for HuggingFace
# Alternative: MODEL_PATH="Tongyi-MAI/Z-Image-Turbo"

# Safe training parameters - optimized for ROCm stability
MAX_EPOCHS=1                # Reduced epochs for faster iteration
LEARNING_RATE=2e-4          # Standard learning rate
NETWORK_DIM=16              # Standard rank
NETWORK_ALPHA=24            # Standard alpha
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4  # Standard accumulation (effective batch size = 4)
GUIDANCE_SCALE=0.0          # Z-Image-Turbo uses 0.0 (no CFG)

# Mixed precision - must match Z-Image weights (bfloat16)
MIXED_PRECISION="bf16"

# ROCm stability optimizations
BLOCKS_TO_SWAP=0            # NO CPU swapping - keep everything on GPU
USE_PINNED_MEMORY_FOR_BLOCK_SWAP=true
VAE_CACHE_CPU=false         # Keep VAE cache on GPU

# Speed optimization settings
DATA_LOADER_WORKERS=0       # Disabled: causes SIGSEGV with data loader workers on ROCm/Strix Halo
PERSISTENT_WORKERS=false     # Disabled: causes SIGSEGV with older accelerate versions on ROCm
SAVE_EVERY_N_EPOCHS=1       # Save less frequently to reduce I/O overhead

# Resume training (set to path of state directory to resume, or leave empty to start fresh)
# Example: RESUME_PATH="outputs/z_image_lora-000001-state"
RESUME_PATH=""              # Leave empty to start new training

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ -n "$RESUME_PATH" ]; then
    echo "RESUMING Z-Image LoRA training..."
    echo "Resuming from: $RESUME_PATH"
else
    echo "Starting Z-Image LoRA training..."
fi
echo "Dataset: $DATASET_CONFIG"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Model: $MODEL_PATH"
echo ""
echo "Speed optimizations enabled:"
echo "  - Reduced epochs: $MAX_EPOCHS (faster iteration)"
echo "  - Reduced gradient accumulation: $GRADIENT_ACCUMULATION_STEPS (faster visible progress)"
echo "  - Data loader workers: $DATA_LOADER_WORKERS (disabled to prevent SIGSEGV on ROCm/Strix Halo)"
echo "  - Persistent workers: disabled (causes SIGSEGV with older accelerate on ROCm)"
echo "  - Save frequency: every $SAVE_EVERY_N_EPOCHS epochs"
echo ""
echo "ROCm stability optimizations enabled:"
echo "  - HSA_XNACK=0: Disabled XNACK for RDNA3.5 stability (no reboot needed)"
echo "  - HSA_ENABLE_SDMA=0: Disabled SDMA to fix allocation faults"
echo "  - HSA_AMD_SDMA_ENABLE=0: Additional SDMA disable (November 2025 fix)"
echo "  - HIP_VISIBLE_DEVICES=0: Force single device to avoid multi-device issues"
echo "  - PYTORCH_ROCM_FORCE_STREAM_ORDERED_ALLOCATOR=1: Use new allocator for List[Tensor] stability"
echo "  - --gradient_checkpointing: Improves stability (not just memory)"
echo "  - --sdpa: Using PyTorch's built-in SDPA (no external dependencies)"
echo "  - Network_dim: $NETWORK_DIM (standard)"
echo "  - Gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - NO CPU swapping: Keeping all data on GPU"
echo ""
echo "NOTE: Using --sdpa (PyTorch built-in) with --split_attn for ROCm stability."
echo "      --split_attn processes attention in smaller chunks to avoid SIGSEGV on Strix Halo."
echo "      HSA_XNACK=0 is set to improve RDNA3.5 stability."
echo ""
echo "System Requirements (recommended for best stability):"
echo "  - Kernel: 6.16.9+ (current: $(uname -r))"
echo "  - ROCm: 7.9+ (current: $(python -c 'import torch; print(torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\")' 2>/dev/null || echo 'check manually'))"
echo "  - Firmware: GC 11.5.1 with MES workaround enabled"
echo "  - Kernel parameter: amdgpu.cwsr_enable=0 (check: cat /proc/cmdline | grep cwsr)"
echo ""
if [ -n "$RESUME_PATH" ]; then
    if [ ! -d "$RESUME_PATH" ]; then
        echo "ERROR: Resume path does not exist: $RESUME_PATH"
        echo "Available checkpoints:"
        ls -d "$OUTPUT_DIR"/*-state* "$OUTPUT_DIR"/*-*-state 2>/dev/null || echo "  (none found)"
        exit 1
    fi
    echo "Resuming from checkpoint: $RESUME_PATH"
    echo ""
fi

# Run training optimized for ROCm stability
# Note: --dit and --vae are set to MODEL_PATH for backward compatibility with common parser
# The actual loading uses --model_path which extracts components from pipeline
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/z_image_train_network.py \
    --model_path "$MODEL_PATH" \
    --dit "$MODEL_PATH" \
    --vae "$MODEL_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    --split_attn \
    $([ "$DISABLE_GRADIENT_CHECKPOINTING" != "false" ] && echo "--gradient_checkpointing") \
    --optimizer_type adamw \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_data_loader_n_workers "$DATA_LOADER_WORKERS" \
    $([ "$PERSISTENT_WORKERS" = "true" ] && echo "--persistent_data_loader_workers") \
    --network_module networks.lora_z_image \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    $([ "$BLOCKS_TO_SWAP" -gt 0 ] && echo "--blocks_to_swap $BLOCKS_TO_SWAP") \
    $([ "$USE_PINNED_MEMORY_FOR_BLOCK_SWAP" = "true" ] && echo "--use_pinned_memory_for_block_swap") \
    $([ "$VAE_CACHE_CPU" = "true" ] && echo "--vae_cache_cpu") \
    --timestep_sampling shift \
    --discrete_flow_shift 5.0 \
    --min_timestep 0 \
    --max_timestep 900 \
    --preserve_distribution_shape \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    --save_state \
    $([ -n "$RESUME_PATH" ] && echo "--resume $RESUME_PATH") \
    --seed 42

echo ""
echo "Training complete! LoRA saved to: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"

