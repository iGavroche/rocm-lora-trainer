# Caching Steps for WAN I2V Training

## Overview

Before training, you need to cache:
1. **Latents** - VAE-encoded video/image latents (saves time during training)
2. **Text Encoder Outputs** - T5-encoded text prompts (saves time during training)

## Review of Caching Scripts

### ✅ `wan_cache_latents.py` - Verified
- **ROCm Compatibility**: ✅ Has ROCm-specific workarounds
  - Disables autocast for VAE encoding on ROCm (prevents zero outputs)
  - Uses float32 for VAE on ROCm instead of bfloat16 (avoids ROCm bfloat16 bugs)
  - Clamps values before dtype conversion (prevents ROCm conversion bugs)
- **I2V Support**: ✅ Handles I2V training correctly
  - Extracts first frame for CLIP encoding
  - Creates image latents for I2V conditioning
  - Handles control images/videos if present
- **Alignment with Training**: ✅ Matches training script
  - Uses same VAE path: `.\models\wan\wan_2.1_vae.safetensors`
  - Uses same dataset config: `dataset.toml`
  - I2V mode enabled (matches `--task i2v-A14B`)

### ✅ `wan_cache_text_encoder_outputs.py` - Verified
- **ROCm Compatibility**: ✅ No special ROCm issues (T5 encoding is straightforward)
- **Alignment with Training**: ✅ Matches training script
  - Uses same T5 path: `.\models\wan\umt5-xxl-enc-bf16.safetensors`
  - Uses same dataset config: `dataset.toml`

### ⚠️ Note on CLIP
- The training script uses `--task i2v-A14B` which indicates I2V training
- I2V training typically requires CLIP for image conditioning
- The cache script has `--clip` argument but training script doesn't specify it
- **If you have CLIP model**, add `--clip .\models\wan\clip_model.safetensors` to the cache script
- **If you don't have CLIP**, the cache will still work but CLIP context will be None

## Steps to Run

### Step 1: Cache Latents

Run the latent caching script:

```powershell
.\cache_latents.ps1
```

Or manually:

```powershell
python wan_cache_latents.py `
  --dataset_config dataset.toml `
  --vae .\models\wan\wan_2.1_vae.safetensors `
  --i2v `
  --batch_size 1 `
  --num_workers 1 `
  --vae_dtype float32 `
  --skip_existing
```

**What it does:**
- Loads VAE model
- Encodes all images/videos in your dataset to latents
- For I2V: Extracts first frame and encodes with CLIP (if provided)
- Saves latents to cache files (one per dataset item)
- Uses float32 for VAE on ROCm to avoid bfloat16 bugs

**Expected output:**
- Cache files saved in your dataset directory (same location as images/videos)
- Files named like `*.latent.safetensors`

### Step 2: Cache Text Encoder Outputs

Run the text encoder caching script:

```powershell
.\cache_text_encoder.ps1
```

Or manually:

```powershell
python wan_cache_text_encoder_outputs.py `
  --dataset_config dataset.toml `
  --t5 .\models\wan\umt5-xxl-enc-bf16.safetensors `
  --batch_size 1 `
  --num_workers 1 `
  --skip_existing
```

**What it does:**
- Loads T5 text encoder model
- Encodes all captions in your dataset
- Saves encoded text to cache files (one per dataset item)
- Removes old cache files not in dataset (unless `--keep_cache` is set)

**Expected output:**
- Cache files saved in your dataset directory
- Files named like `*.text_encoder_output.safetensors`

### Step 3: Verify Cache Files

You can verify cache files were created:

```powershell
python check_cache.py --dataset_config dataset.toml
```

## Important Notes

1. **Windows + ROCm**: Use `--num_workers 1` (ThreadPoolExecutor uses threads, not multiprocessing, so 1 worker is safe)

2. **VAE Dtype**: The script automatically uses `float32` for VAE on ROCm to avoid bfloat16 bugs. This is correct.

3. **Skip Existing**: Both scripts use `--skip_existing` to skip already-cached files. Remove this flag if you want to re-cache everything.

4. **Batch Size**: Set to 1 for stability. You can increase if you have more VRAM, but 1 is safest.

5. **Order**: You can run these in any order, but typically:
   - Cache latents first (takes longer)
   - Cache text encoder outputs second (faster)

6. **Re-caching**: If you change your dataset or want to re-cache:
   - Remove `--skip_existing` flag
   - Or delete the cache files manually

## Troubleshooting

### VAE encoding produces zeros
- **Cause**: ROCm bfloat16 bug
- **Fix**: Script already uses float32 for VAE on ROCm (automatic)

### Cache files not found during training
- **Cause**: Cache files not created or in wrong location
- **Fix**: Run caching scripts again, check dataset paths

### Out of memory during caching
- **Fix**: Reduce batch size to 1, or use `--vae_cache_cpu` to cache VAE features on CPU

### CLIP not found (I2V training)
- **Cause**: CLIP model not provided
- **Fix**: If you have CLIP model, add `--clip` argument to cache script
- **Note**: Training may still work without CLIP, but image conditioning won't be available

