# Generate Video with Chani LoRA using WAN 2.2

This guide explains how to generate videos with your trained chani LoRA using musubi-tuner.

## Overview

- **LoRA Model**: Trained with musubi-tuner (located at `/home/nino/projects/wantraining/comfyui_lora/chani.safetensors`)
- **Base Model**: WAN 2.2 Ti2V 14B (note: musubi-tuner supports 14B models, not 5B)
- **Compatibility**: The LoRA is in ComfyUI format and is compatible with ComfyUI

## Quick Start

Run the generation script:

```bash
./generate_chani_video.sh
```

This will:
1. Load the WAN 2.2 low-noise and high-noise models
2. Apply the chani LoRA with multiplier 1.0
3. Generate a 81-frame video at 512x512 resolution
4. Save output as both video file and individual frames

Output will be saved to `outputs/chani_output.mp4`

## Manual Generation

If you want to customize the generation, run:

```bash
python src/musubi_tuner/wan_generate_video.py \
    --task t2v-A14B \
    --dit models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --dit_high_noise models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
    --vae models/wan/wan_2.1_vae.safetensors \
    --t5 models/wan/umt5-xxl-enc-bf16.safetensors \
    --lora_weight /home/nino/projects/wantraining/comfyui_lora/chani.safetensors \
    --lora_multiplier 1.0 \
    --video_size 512 512 \
    --video_length 81 \
    --fps 16 \
    --infer_steps 20 \
    --prompt "Chani, 32 year old woman, blond with blue eyes, beautiful, calm expression, wavy hair, pretty wear, normal lighting" \
    --guidance_scale 5.0 \
    --save_path outputs/chani_output.mp4 \
    --output_type both \
    --attn_mode torch \
    --fp8 \
    --timestep_boundary 0.875 \
    --cpu_noise
```

## Customization Options

### Change Video Dimensions
Modify `--video_size WIDTH HEIGHT` (e.g., `--video_size 832 480`)

### Change Video Length
Modify `--video_length FRAMES` (e.g., `--video_length 49`)

### Change Inference Steps
Modify `--infer_steps NUMBER` (e.g., `--infer_steps 30` for higher quality, slower)

### Adjust LoRA Strength
Modify `--lora_multiplier VALUE` (e.g., `1.5` for stronger effect, `0.5` for weaker)

### Change Prompt
Modify `--prompt "YOUR PROMPT HERE"`

### Use Different Attention Mode
- `--attn_mode torch` (default, no installation needed)
- `--attn_mode sdpa` (same as torch)
- `--attn_mode flash2` (requires flash-attn installation)
- `--attn_mode xformers` (requires xformers installation)

### Memory Optimization Options
- `--fp8` - Use fp8 mode (reduces memory)
- `--fp8_t5` - Use fp8 for T5 encoder (further reduces memory)
- `--vae_cache_cpu` - Cache VAE on CPU

### Generation Speed vs Quality
- **Fast**: `--infer_steps 10 --cfg_skip_mode early --cfg_apply_ratio 0.5`
- **Balanced**: `--infer_steps 20` (default)
- **High Quality**: `--infer_steps 40 --fp8` (no fp8_scaled)

## Batch Generation

To generate multiple videos, create a prompts file:

```bash
# Create prompts.txt
cat > prompts.txt << EOF
Chani, 32 year old woman, blond with blue eyes, beautiful, calm expression, wavy hair, pretty wear, normal lighting
Chani walking through a garden --w 832 --h 480 --f 49
Chani in a modern apartment --w 512 --h 768
EOF

# Run batch generation
python src/musubi_tuner/wan_generate_video.py \
    --from_file prompts.txt \
    --task t2v-A14B \
    --dit models/wan/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --dit_high_noise models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
    --vae models/wan/wan_2.1_vae.safetensors \
    --t5 models/wan/umt5-xxl-enc-bf16.safetensors \
    --lora_weight /home/nino/projects/wantraining/comfyui_lora/chani.safetensors \
    --save_path outputs
```

## Troubleshooting

### Out of Memory Error
Try:
- Add `--fp8` flag
- Add `--fp8_t5` flag
- Reduce video size: `--video_size 256 256`
- Reduce steps: `--infer_steps 10`

### Model Not Found
Ensure all model files are in `models/wan/`:
- `umt5-xxl-enc-bf16.safetensors`
- `wan_2.1_vae.safetensors`
- `wan2.2_t2v_low_noise_14B_fp16.safetensors`
- `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`

### LoRA Not Working
Check:
- LoRA path is correct
- LoRA format is compatible (sd-scripts format)
- `--lora_multiplier` is set appropriately

## Notes

- The WAN 2.2 architecture in musubi-tuner supports **14B models**, not 5B models
- Your chani LoRA is trained with musubi-tuner and is already in ComfyUI-compatible format
- The `--cpu_noise` flag ensures same results as ComfyUI for the same seed
- The LoRA multiplier controls how strongly the LoRA affects the generation (1.0 = full strength)

## Using in ComfyUI

Your LoRA file at `/home/nino/projects/wantraining/comfyui_lora/chani.safetensors` can be directly used in ComfyUI:

1. Copy the file to your ComfyUI models/loras directory
2. Use any WAN 2.2 Ti2V node in ComfyUI
3. Add the LoRA node and select your chani LoRA
4. Set the multiplier (e.g., 1.0 for full strength)








