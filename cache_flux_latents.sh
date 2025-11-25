#!/bin/bash
# Cache VAE latents for FLUX training

source .venv/bin/activate

DATASET_CONFIG="dataset_flux.toml"
VAE_PATH="/home/nino/ComfyUI/models/vae/ae.safetensors"

echo "Caching FLUX VAE latents..."
echo "VAE: $VAE_PATH"
echo ""

python src/musubi_tuner/flux_kontext_cache_latents.py \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE_PATH"

echo "VAE latent caching complete!"



