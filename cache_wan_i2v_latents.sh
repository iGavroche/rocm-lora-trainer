#!/bin/bash
# Cache VAE latents for WAN 2.2 I2V training
# Note: WAN 2.2 I2V does NOT require CLIP (unlike WAN 2.1)

source .venv/bin/activate

DATASET_CONFIG="dataset.toml"
VAE_PATH="models/wan/wan_2.1_vae.safetensors"

echo "Caching WAN 2.2 I2V VAE latents..."
echo "Dataset: $DATASET_CONFIG"
echo "VAE: $VAE_PATH"
echo ""
echo "Note: This will cache image latents needed for I2V training."
echo "WAN 2.2 I2V does NOT require CLIP model."
echo ""

python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE_PATH" \
    --i2v \
    --batch_size 1

echo ""
echo "VAE latent caching complete!"
echo ""
echo "Next, cache text encoder outputs if not already done:"
echo "  ./cache_text_encoder.sh"









