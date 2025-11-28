#!/bin/bash
# Train chani LoRA using WAN 2.2 T2V with MINIMAL settings for proof of concept
# This uses lower quality settings to minimize computation while still getting results

# Activate virtual environment
source .venv/bin/activate

# Configuration
DATASET_CONFIG="dataset_minimal.toml"
OUTPUT_DIR="outputs"
OUTPUT_NAME="chani_minimal"

# Model paths
VAE_PATH="models/wan/wan_2.1_vae.safetensors"
T5_PATH="models/wan/umt5-xxl-enc-bf16.safetensors"
DIT_LOW_NOISE="models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors"
# DIT_HIGH_NOISE not needed for minimal training

# Minimal training parameters for proof of concept
MAX_EPOCHS=2           # Minimal epochs (proof of concept)
LEARNING_RATE=1e-4     # Lower learning rate for stability
NETWORK_DIM=16         # Lower rank (minimal)
NETWORK_ALPHA=8        # Lower alpha
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4  # Smaller effective batch size
GUIDANCE_SCALE=1.0

# Mixed precision - must match DiT weights (fp16)
MIXED_PRECISION="fp16"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting MINIMAL WAN 2.2 LoRA training for chani (Proof of Concept)..."
echo "Dataset: $DATASET_CONFIG"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""
echo "NOTE: This uses minimal settings for quick proof of concept:"
echo "  - Only 2 epochs"
echo "  - Low network rank (16)"
echo "  - Lower learning rate"
echo "  - FP8 mode for memory efficiency"
echo ""
echo "This will produce a basic working LoRA quickly to validate the setup."

# Run training with accelerate - using fp8 for minimal VRAM usage
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    --optimizer_type adamw \
    --learning_rate "$LEARNING_RATE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_data_loader_n_workers 1 \
    --network_module networks.lora_wan \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0 \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME"

echo ""
echo "Minimal training complete!"
echo "LoRA saved to $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo ""
echo "You can now test this LoRA with generate_chani_video.sh"
echo "For better quality, use train_chani.sh with higher settings"


