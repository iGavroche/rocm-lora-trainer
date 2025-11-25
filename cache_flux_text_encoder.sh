#!/bin/bash
# Cache T5 and CLIP text encoder outputs for FLUX training

source .venv/bin/activate

DATASET_CONFIG="dataset_flux.toml"
T5_PATH="/home/nino/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors"
CLIP_PATH="/home/nino/ComfyUI/models/text_encoders/clip_l.safetensors"

echo "Caching FLUX text encoder outputs..."
echo "T5: $T5_PATH"
echo "CLIP: $CLIP_PATH"
echo ""

python src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_CONFIG" \
    --text_encoder1 "$T5_PATH" \
    --text_encoder2 "$CLIP_PATH" \
    --batch_size 8

echo "Text encoder caching complete!"



