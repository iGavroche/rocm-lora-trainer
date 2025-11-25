#!/bin/bash
# Train chani LoRA using WAN 2.2 I2V with SAFE settings to prevent driver crashes
# Using only low-noise model to reduce memory pressure on ROCm

# Activate virtual environment
source .venv/bin/activate

# ROCm stability: Disable XNACK for RDNA3.5 (Strix Halo)
# XNACK can cause instability with ROCm on RDNA3.5
export HSA_XNACK=0

# Configuration
DATASET_CONFIG="dataset.toml"  # Using full dataset config with 5 repeats
OUTPUT_DIR="outputs"
OUTPUT_NAME="chani_full"

# Model paths
VAE_PATH="models/wan/wan_2.1_vae.safetensors"
T5_PATH="models/wan/umt5-xxl-enc-bf16.safetensors"
DIT_LOW_NOISE="models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors"
# Using ONLY low-noise model to prevent driver crashes
# High-noise model requires too much VRAM and causes driver crashes on ROCm

# Safe training parameters - optimized for ROCm stability (NOT memory - you have 96GB VRAM!)
MAX_EPOCHS=2                # Reduced epochs for faster iteration (was 6)
LEARNING_RATE=2e-4          # Standard learning rate
NETWORK_DIM=16              # Standard rank (you have 96GB VRAM, no need to reduce)
NETWORK_ALPHA=32            # Standard alpha
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4  # Standard accumulation (effective batch size = 4)
GUIDANCE_SCALE=1.0

# Mixed precision - must match DiT weights (fp16)
MIXED_PRECISION="fp16"

# ROCm stability optimizations (NOT memory - the crashes are permission faults, not OOM)
# With 96GB VRAM, keep everything on GPU to avoid driver issues
BLOCKS_TO_SWAP=0            # NO CPU swapping - keep everything on GPU (you have 96GB!)
VAE_CACHE_CPU=false         # Keep VAE cache on GPU

# Speed optimization settings
DATA_LOADER_WORKERS=4        # Increase data loading speed (default is 8, but 4 is safer)
PERSISTENT_WORKERS=true      # Keep workers alive between epochs (faster epoch transitions)
SAVE_EVERY_N_EPOCHS=1       # Save less frequently to reduce I/O overhead

# Resume training (set to path of state directory to resume, or leave empty to start fresh)
# Example: RESUME_PATH="outputs/chani_full-000001-state"
# To find available checkpoints: ls -d outputs/*-state* outputs/*-*-state
RESUME_PATH=""              # Leave empty to start new training

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ -n "$RESUME_PATH" ]; then
    echo "RESUMING WAN 2.2 I2V LoRA training for chani..."
    echo "Resuming from: $RESUME_PATH"
else
    echo "Starting OPTIMIZED WAN 2.2 I2V LoRA training for chani..."
fi
echo "Dataset: $DATASET_CONFIG (48 images, 5 repeats = 240 steps per epoch)"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Using ONLY low-noise model (safer for ROCm, prevents driver crashes)"
echo ""
echo "Speed optimizations enabled:"
echo "  - Reduced epochs: $MAX_EPOCHS (faster iteration)"
echo "  - Reduced gradient accumulation: $GRADIENT_ACCUMULATION_STEPS (faster visible progress)"
echo "  - Data loader workers: $DATA_LOADER_WORKERS"
echo "  - Persistent workers: enabled (faster epoch transitions)"
echo "  - Save frequency: every $SAVE_EVERY_N_EPOCHS epochs"
echo ""
echo "ROCm stability optimizations enabled:"
echo "  - HSA_XNACK=0: Disabled XNACK for RDNA3.5 stability (no reboot needed)"
echo "  - --gradient_checkpointing: Improves stability (not just memory)"
echo "  - --sdpa: Using PyTorch's built-in SDPA (no external dependencies)"
echo "  - Network_dim: $NETWORK_DIM (standard, you have 96GB VRAM)"
echo "  - Gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - NO CPU swapping: Keeping all data on GPU (96GB available)"
echo ""
echo "NOTE: Using --sdpa (PyTorch built-in) since xformers is not installed."
echo "      HSA_XNACK=0 is set to improve RDNA3.5 stability."
echo ""
echo "⚠️  For additional stability, consider applying kernel parameters:"
echo "      Run: ./apply_rocm_kernel_params.sh (requires sudo + reboot)"
echo "      This adds: amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.isolation=0"
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
echo "NOTE: This uses only the low-noise model to prevent driver crashes."
echo "      Quality will still be good, but slightly lower than dual-model training."
echo ""

# Run training optimized for ROCm stability
# With 96GB VRAM, we keep everything on GPU to avoid driver permission faults
# Note: --save_state is needed to save checkpoints that can be resumed
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/wan_train_network.py \
    --task i2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    $([ "$DISABLE_GRADIENT_CHECKPOINTING" = "false" ] && echo "--gradient_checkpointing") \
    --optimizer_type adamw \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_data_loader_n_workers "$DATA_LOADER_WORKERS" \
    $([ "$PERSISTENT_WORKERS" = "true" ] && echo "--persistent_data_loader_workers") \
    --network_module networks.lora_wan \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    $([ "$BLOCKS_TO_SWAP" -gt 0 ] && echo "--blocks_to_swap $BLOCKS_TO_SWAP") \
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

