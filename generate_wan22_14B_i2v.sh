#!/bin/bash
# Generate video with WAN 2.2 T2V LoRA
# Note: Using musubi-tuner (not diffsynth)

# Activate virtual environment
source .venv/bin/activate

# Set paths
LORA_PATH="/home/nino/projects/musubi-tuner/outputs/wan22_14B_t2v_minimal.safetensors"
T5_PATH="/home/nino/projects/musubi-tuner/models/wan/umt5-xxl-enc-bf16.safetensors"
VAE_PATH="/home/nino/projects/musubi-tuner/models/wan/wan_2.1_vae.safetensors"
DIT_LOW_NOISE="/home/nino/projects/musubi-tuner/models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="/home/nino/projects/musubi-tuner/models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"

# Output directory
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

echo "Generating video with WAN 2.2 LoRA using musubi-tuner..."
echo "LoRA: $LORA_PATH"

# Generate video using WAN 2.2 14B model
python src/musubi_tuner/wan_generate_video.py \
    --task t2v-A14B \
    --dit "$DIT_LOW_NOISE" \
    --sample_solver vanilla \
    --vae "$VAE_PATH" \
    --t5 "$T5_PATH" \
    --lora_weight "$LORA_PATH" \
    --lora_multiplier 1.0 \
    --video_size 480 832 \
    --video_length 9 \
    --fps 8 \
    --infer_steps 10 \
    --prompt "A person, beautiful, calm expression, wavy hair, pretty wear, normal lighting" \
    --guidance_scale 5.0 \
    --save_path "$OUTPUT_DIR/wan22_14B_i2v_output.mp4" \
    --output_type both \
    --attn_mode torch

echo "Video generation complete. Output saved to $OUTPUT_DIR/wan22_14B_i2v_output.mp4"
echo "This LoRA is trained with musubi and is compatible with ComfyUI format."

