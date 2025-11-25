#!/bin/bash
# Simple video generation with GPU memory cleanup

source .venv/bin/activate

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Simple generation with minimal settings
python src/musubi_tuner/wan_generate_video.py \
    --task t2v-A14B \
    --dit models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --vae models/wan/wan_2.1_vae.safetensors \
    --t5 models/wan/umt5-xxl-enc-bf16.safetensors \
    --lora_weight outputs/chani_minimal.safetensors \
    --lora_multiplier 1.0 \
    --video_size 480 832 \
    --video_length 5 \
    --fps 8 \
    --infer_steps 8 \
    --sample_solver vanilla \
    --prompt "a woman with blond hair" \
    --guidance_scale 3.0 \
    --save_path outputs/test_chani.mp4 \
    --output_type video \
    --attn_mode torch \
    --seed 42

echo "Generation attempt complete."





