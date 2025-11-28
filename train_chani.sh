#!/bin/bash
# Train chani LoRA using WAN 2.2 T2V with cached latents
# This script trains a LoRA on the myface dataset using cached latents

# Configuration
DATASET_CONFIG="dataset.toml"
OUTPUT_DIR="outputs"
OUTPUT_NAME="chani"

# Model paths
VAE_PATH="models/wan/wan_2.1_vae.safetensors"
T5_PATH="models/wan/umt5-xxl-enc-bf16.safetensors"
DIT_LOW_NOISE="models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"

# Training parameters
MAX_EPOCHS=16
LEARNING_RATE=2e-4
NETWORK_DIM=32
NETWORK_ALPHA=16
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 8
GUIDANCE_SCALE=1.0

# Mixed precision
MIXED_PRECISION="bf16"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting WAN 2.2 LoRA training for chani..."
echo "Dataset: $DATASET_CONFIG"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"

# Run training with accelerate
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate "$LEARNING_RATE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_wan \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME"

echo "Training complete. LoRA saved to $OUTPUT_DIR/$OUTPUT_NAME.safetensors"


