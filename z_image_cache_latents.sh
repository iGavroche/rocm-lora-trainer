#!/bin/bash
# Cache VAE latents for Z-Image training
# This script must be run before training to create latent caches

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
DATASET_CONFIG="dataset.toml"

# Model path: will auto-detect local files in models/z-image/, or download from HuggingFace
# If you have local files (ae.safetensors, z_image_turbo_bf16.safetensors, qwen_3_4b.safetensors),
# they will be used automatically. Otherwise, will download from HuggingFace.
MODEL_PATH="models/z-image"  # Local model directory (auto-detects files), or use "Tongyi-MAI/Z-Image-Turbo" for HuggingFace only

# Cache settings
BATCH_SIZE=1  # Process one image at a time
NUM_WORKERS=0  # Disabled for ROCm stability
SKIP_EXISTING=true  # Skip already cached files
DEVICE="cuda"  # Use "cuda" for GPU, "cpu" for CPU

echo "Z-Image Latent Cache Creation"
echo "============================="
echo "Dataset: $DATASET_CONFIG"
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Skip existing: $SKIP_EXISTING"
echo ""

# Run cache script
python src/musubi_tuner/z_image_cache_latents.py \
    --model_path "$MODEL_PATH" \
    --vae "$MODEL_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    $([ "$SKIP_EXISTING" = "true" ] && echo "--skip_existing")

if [ $? -eq 0 ]; then
    echo ""
    echo "Latent cache creation complete!"
    echo "You can now run ./z_image_cache_text_encoder_outputs.sh to cache text encoder outputs."
else
    echo ""
    echo "Latent cache creation failed!"
    exit 1
fi

