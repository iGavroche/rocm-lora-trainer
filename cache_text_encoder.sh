#!/bin/bash
# Cache T5 text encoder outputs for WAN training

source .venv/bin/activate

echo "Caching T5 text encoder outputs for WAN training..."

python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config dataset.toml \
    --t5 models/wan/umt5-xxl-enc-bf16.safetensors \
    --batch_size 16

echo "Text encoder caching complete!"






