# Training Summary - Chani LoRA

## Current Status

You have two working training setups:

### ✅ 1. WAN 2.2 (Video Training) - WORKING
**Status:** ✅ Fully working, trained successfully  
**Location:** `outputs/chani_minimal.safetensors` (2 epochs proof of concept)

#### What worked:
- Training completed successfully (13 minutes)
- Using: `train_chani_minimal.sh`
- Optimizations for ROCm:
  - Use `adamw` (not `adamw8bit`) 
  - Use `fp16` mixed precision (not `bf16`)
  - Remove `--fp8_base` flag
  - Remove `--gradient_checkpointing_cpu_offload`

#### Available training scripts:
- `train_chani_minimal.sh` - 2 epochs, ~13 min ✅ PROOF OF CONCEPT DONE
- `train_chani_fast.sh` - 3 epochs, ~3 hours
- `train_chani_full.sh` - 6 epochs, ~24 hours

### ❌ 2. FLUX Kontext (Image Training) - NOT WORKING
**Status:** ❌ Incompatible model format  
**Issue:** Your FLUX models are FP8-scaled format, musubi-tuner expects standard FP16/BF16

**Why it failed:**
- `flux1-dev-kontext_fp8_scaled.safetensors` has FP8 scaling weights
- musubi-tuner's FLUX loader expects standard FP16/BF16
- No non-FP8 FLUX Kontext model available

## Recommendation

**Use WAN 2.2 for training** since it:
- ✅ Works with your current setup
- ✅ Already successfully trained
- ✅ Faster than expected (9s/step)
- ✅ ROCm compatible

### For better quality with WAN:
Run with more epochs:
```bash
./train_chani_fast.sh  # 3 epochs, ~3 hours, good balance
# OR
./train_chani_full.sh  # 6 epochs, ~24 hours, best quality
```

### Why WAN video training works for faces:
1. Even though it's "video" training, you can still use it for image generation
2. The LoRA learns the facial features from your dataset
3. Works better than trying to force FLUX Kontext

## Dataset Status

✅ **Dataset is ready:**
- 48 images at 512x512
- Captions consistent
- VAE latents cached for WAN
- Text encoder (T5) cached for WAN

**Training time breakdown per epoch:**
- 240 steps (48 images × 5 repeats)
- ~9 seconds per step
- Total: ~36 minutes per epoch

## Next Steps

1. **Quick test:** Continue with `chani_minimal.safetensors` (already done)
2. **Better quality:** Run `./train_chani_fast.sh` (3 epochs, ~3 hours)
3. **Best quality:** Run `./train_chani_full.sh` (6 epochs, ~24 hours)

## Note on FLUX

If you want to use FLUX in the future, you'll need to download the **standard FP16/BF16** versions from HuggingFace, not the FP8-scaled versions that ComfyUI uses.



