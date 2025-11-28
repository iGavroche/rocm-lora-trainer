#!/bin/bash
# Train FLUX.1 Kontext LoRA with minimal settings for fast iteration

source .venv/bin/activate

DATASET_CONFIG="dataset_flux.toml"
OUTPUT_DIR="outputs"
OUTPUT_NAME="flux_kontext"

# Model paths - copied from ComfyUI
DIT_PATH="/home/nino/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"
VAE_PATH="/home/nino/ComfyUI/models/vae/ae.safetensors"
T5_PATH="/home/nino/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors"
CLIP_PATH="/home/nino/ComfyUI/models/text_encoders/clip_l.safetensors"

# Fast training parameters
MAX_EPOCHS=4                 # 4 epochs for good balance
LEARNING_RATE=1e-4
NETWORK_DIM=24               # Lower rank for speed
NETWORK_ALPHA=12
GRADIENT_ACCUMULATION_STEPS=4

MIXED_PRECISION="bf16"       # FLUX recommends bf16

mkdir -p "$OUTPUT_DIR"

echo "Starting FLUX Kontext LoRA training..."
echo "Dataset: $DATASET_CONFIG"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo "Epochs: 4 (estimated time: ~2-3 hours)"
echo ""

# First check if we need to copy models
if [ ! -f "models/flux/flux1-dev-kontext_fp8_scaled.safetensors" ]; then
    echo "Copying FLUX models..."
    mkdir -p models/flux
    cp "$DIT_PATH" models/flux/
    echo "DiT model copied"
fi

if [ ! -f "models/flux/ae.safetensors" ]; then
    cp "$VAE_PATH" models/flux/
    echo "VAE model copied"
fi

if [ ! -f "models/flux/t5xxl_fp16.safetensors" ]; then
    cp "$T5_PATH" models/flux/
    echo "T5 model copied"
fi

if [ ! -f "models/flux/clip_l.safetensors" ]; then
    cp "$CLIP_PATH" models/flux/
    echo "CLIP model copied"
fi

echo ""
echo "Models ready, starting training..."
echo ""

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
    src/musubi_tuner/flux_kontext_train_network.py \
    --dit models/flux/flux1-dev-kontext_fp8_scaled.safetensors \
    --vae models/flux/ae.safetensors \
    --text_encoder1 models/flux/t5xxl_fp16.safetensors \
    --text_encoder2 models/flux/clip_l.safetensors \
    --dataset_config "$DATASET_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --mixed_precision "$MIXED_PRECISION" \
    --sdpa \
    --optimizer_type adamw \
    --learning_rate "$LEARNING_RATE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --network_module networks.lora_flux \
    --network_dim "$NETWORK_DIM" \
    --network_alpha "$NETWORK_ALPHA" \
    --timestep_sampling flux_shift \
    --weighting_scheme none \
    --max_train_epochs "$MAX_EPOCHS" \
    --save_every_n_epochs 2 \
    --seed 42

echo ""
echo "FLUX training complete! LoRA saved to: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"

