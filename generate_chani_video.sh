#!/bin/bash
# Generate video with chani LoRA using WAN 2.2 I2V
# NOTE: This LoRA was trained with --task i2v-A14B, so we MUST use I2V task here!
# I2V requires an input image, so we use one from the training dataset

# Activate virtual environment
source .venv/bin/activate

# Set paths
LORA_PATH="/home/nino/projects/musubi-tuner/outputs/chani_full.safetensors"  # Using full training LoRA
T5_PATH="/home/nino/projects/musubi-tuner/models/wan/umt5-xxl-enc-bf16.safetensors"
VAE_PATH="/home/nino/projects/musubi-tuner/models/wan/wan_2.1_vae.safetensors"
DIT_LOW_NOISE="/home/nino/projects/musubi-tuner/models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="/home/nino/projects/musubi-tuner/models/wan/wan2.2_i2v_high_noise_14B_fp16.safetensors"

# Input image (using one from training dataset)
IMAGE_PATH="myface/image0001.jpg"  # Use any image from your training dataset

# Output directory
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

echo "Generating video with chani LoRA using musubi-tuner (I2V)..."
echo "LoRA: $LORA_PATH"
echo "Input image: $IMAGE_PATH"
echo ""
echo "IMPORTANT: This LoRA was trained with --task i2v-A14B, so we MUST use I2V task!"
echo "If you want T2V generation, you need to retrain with --task t2v-A14B"
echo ""

# Generate video with chani character using WAN 2.2 I2V model
python src/musubi_tuner/wan_generate_video.py \
    --task i2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --image_path "$IMAGE_PATH" \
    --sample_solver vanilla \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --lora_weight "$LORA_PATH" \
    --lora_multiplier 1.0 \
    --video_size 480 832 \
    --video_length 9 \
    --fps 8 \
    --infer_steps 10 \
    --prompt "Chani, 32 year old woman, blond with blue eyes, beautiful, calm expression, wavy hair, pretty wear, normal lighting" \
    --guidance_scale 5.0 \
    --save_path "$OUTPUT_DIR/chani_i2v_output.mp4" \
    --output_type both \
    --sdpa

echo ""
echo "Video generation complete. Output saved to $OUTPUT_DIR/chani_i2v_output.mp4"
echo "This LoRA is trained with musubi and is compatible with ComfyUI format."

