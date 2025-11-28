# LoRA Troubleshooting Guide

## Issue: LoRA Not Affecting Output

### Root Cause: Task Mismatch

**Problem**: The LoRA was trained with `--task i2v-A14B` (Image-to-Video) but the generation script was using `--task t2v-A14B` (Text-to-Video).

**Why this matters**: I2V and T2V have different model architectures:
- **I2V (Image-to-Video)**: Requires an input image and generates video from that image
- **T2V (Text-to-Video)**: Only uses text prompts, no input image

The LoRA weights are task-specific and won't work correctly if the task doesn't match.

### Solution

**Option 1: Use I2V Generation (Current Fix)**
- Updated `generate_chani_video.sh` to use `--task i2v-A14B`
- Added `--image_path` argument pointing to a training image
- Changed model paths to I2V models (`wan2.2_i2v_low_noise_14B_fp16.safetensors`)
- Updated LoRA path to `chani_full.safetensors` (the I2V-trained LoRA)

**Option 2: Retrain for T2V**
If you prefer text-to-video generation:
1. Update `train_chani_full.sh` to use `--task t2v-A14B`
2. Change model paths to T2V models
3. Remove I2V-specific arguments (`--min_timestep`, `--max_timestep`, `--preserve_distribution_shape`)
4. Retrain the LoRA

### Driver Issues Check

No driver crashes were found in the training logs. The training completed successfully and produced a valid LoRA file (1200 keys, 147MB).

### Verification

To verify the LoRA is working:
1. Generate with the updated script: `./generate_chani_video.sh`
2. Compare output with and without the LoRA (remove `--lora_weight` argument)
3. The LoRA should now affect the output since task matches

### Files Updated

- `generate_chani_video.sh`: Fixed to use I2V task matching the training
- `train_chani_full.sh`: Already correctly configured for I2V training

