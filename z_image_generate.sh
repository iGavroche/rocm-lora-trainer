#!/bin/bash
# Generate images using Z-Image with optional LoRA weights

# Activate virtual environment
source .venv/bin/activate

# ROCm stability settings (same as training)
export HSA_XNACK=0
export HSA_ENABLE_SDMA=0
export PYTORCH_ROCM_ARCH=gfx1151
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HIP_FORCE_DEV_KERNELS=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
MODEL_PATH="models/z-image"  # Local model directory, or use "Tongyi-MAI/Z-Image-Turbo" for HuggingFace
OUTPUT_DIR="outputs/generated"
PROMPT="A beautiful landscape with mountains and a lake"
NEGATIVE_PROMPT=""  # Optional negative prompt
HEIGHT=1024
WIDTH=1024
NUM_STEPS=8  # Z-Image-Turbo uses 8 steps
GUIDANCE_SCALE=0.0  # Z-Image-Turbo uses 0.0 (no CFG)
SEED=42  # Set to null for random seed

# LoRA settings (optional)
LORA_WEIGHT=""  # Path to LoRA safetensors file, e.g., "outputs/z_image_lora.safetensors"
LORA_MULTIPLIER=1.0  # LoRA strength multiplier

# Device
DEVICE="cuda"  # Use "cuda" for GPU, "cpu" for CPU

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Z-Image Generation"
echo "=================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
if [ -n "$NEGATIVE_PROMPT" ]; then
    echo "Negative prompt: $NEGATIVE_PROMPT"
fi
echo "Image size: ${HEIGHT}x${WIDTH}"
echo "Steps: $NUM_STEPS"
echo "Guidance scale: $GUIDANCE_SCALE"
if [ -n "$LORA_WEIGHT" ]; then
    echo "LoRA: $LORA_WEIGHT (multiplier: $LORA_MULTIPLIER)"
fi
echo "Output directory: $OUTPUT_DIR"
echo ""

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/z_image_${TIMESTAMP}.png"

# Build command
CMD="python src/musubi_tuner/z_image_generate_image.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --prompt \"$PROMPT\""
if [ -n "$NEGATIVE_PROMPT" ]; then
    CMD="$CMD --negative_prompt \"$NEGATIVE_PROMPT\""
fi
CMD="$CMD --image_size $HEIGHT $WIDTH"
CMD="$CMD --num_inference_steps $NUM_STEPS"
CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
if [ -n "$SEED" ] && [ "$SEED" != "null" ]; then
    CMD="$CMD --seed $SEED"
fi
if [ -n "$LORA_WEIGHT" ]; then
    CMD="$CMD --lora_weight \"$LORA_WEIGHT\""
    CMD="$CMD --lora_multiplier $LORA_MULTIPLIER"
fi
CMD="$CMD --device \"$DEVICE\""
CMD="$CMD --output \"$OUTPUT_FILE\""

# Optional flags
if [ "$SPLIT_ATTN" = "true" ]; then
    CMD="$CMD --split_attn"
fi

echo "Running: $CMD"
echo ""

# Execute
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Generation complete! Image saved to: $OUTPUT_FILE"
else
    echo ""
    echo "Generation failed!"
    exit 1
fi


