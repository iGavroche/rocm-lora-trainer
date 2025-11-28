#!/bin/bash
# Train chani LoRA - FAST version with reduced compute
# Reduced epochs and optimized settings for speed

source .venv/bin/activate

DATASET_CONFIG="dataset.toml"
OUTPUT_DIR="outputs"
OUTPUT_NAME="chani_fast"

VAE_PATH="models/wan/wan_2.1_vae.safetensors"
T5_PATH="models/wan/umt5-xxl-enc-bf16.safetensors"
DIT_LOW_NOISE="models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors"

# Fast training parameters - optimized for speed
MAX_EPOCHS=3                 # Only 3 epochs for faster training
LEARNING_RATE=2e-4
NETWORK_DIM=24               # Slightly reduced rank
NETWORK_ALPHA=12
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=12  # Fewer updates, faster
GUIDANCE_SCALE=1.0

MIXED_PRECISION="fp16"

mkdir -p "$OUTPUT_DIR"

echo "Starting FAST WAN 2.2 LoRA training for chani..."
echo "Dataset: $DATASET_CONFIG (48 images, 5 repeats = 240 steps per epoch)"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Epochs: 3 (estimated time: ~3 hours total)"
echo ""

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    --optimizer_type adamw \
    --learning_rate "$LEARNING_RATE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --network_module networks.lora_wan \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0 \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs 1 \
    --seed 42

echo ""
echo "Fast training complete! LoRA saved to: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"




