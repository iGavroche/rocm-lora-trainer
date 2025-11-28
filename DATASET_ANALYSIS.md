# Training Dataset Analysis for Chani LoRA

## Dataset Overview ✅

**Status: GOOD for training**

### Image Quality
- **Count:** 48 images (all properly paired with captions)
- **Resolution:** 512x512 (consistent across all images)
- **Format:** JPEG
- **File sizes:** 42KB - 62KB per image (good quality)
- **Aspect ratio:** 1:1 (square format)

### Captions
All images have proper .txt caption files with consistent format:
```
Chani, 32 year old woman, [expression], [hair], [clothing], [lighting]
```

**Good points:**
- Consistent subject identity ("Chani, 32 year old woman")
- Varied expressions (calm, neutral, gentle smile, serene)
- Varied hair descriptions (wavy, long, medium length, shoulder length)
- Varied clothing (casual outfit, everyday clothing, simple wear, dark clothing)
- Consistent lighting (low lighting throughout)

### Cached Data ✅
Both cache types are present and verified:
- ✅ VAE latents: `image0001_0512x0512_wan.safetensors` (129KB each)
- ✅ Text encoder: `image0001_wan_te.safetensors` (185KB each)
- ✅ All 48 images have both cache files

## Recommendations for Production Training

### Current Configuration (dataset.toml)
```
num_repeats = 5          # 240 steps per epoch (48 images × 5)
batch_size = 1
resolution = [512, 512]
enable_bucket = true
```

### Training Settings Recommendation

**Option 1: Quick Quality (train_chani_minimal.sh)**
- Epochs: 2
- Rank: 16
- Learning rate: 1e-4
- Time: ~13 minutes
- Quality: ⭐⭐ (proof of concept)

**Option 2: Balanced (recommended)**
- Epochs: 6
- Rank: 32
- Learning rate: 2e-4
- Time: ~40 minutes
- Quality: ⭐⭐⭐⭐ (good balance)

**Option 3: Full Quality (train_chani_full.sh)**
- Epochs: 10
- Rank: 32
- Learning rate: 2e-4
- Time: ~65 minutes
- Quality: ⭐⭐⭐⭐⭐ (production)

## Issues Identified ⚠️

1. **Limited diversity in lighting**: All images have "low lighting" - consider adding brighter lighting examples for better generalization

2. **Limited clothing variety**: Most are "casual outfit", "everyday clothing", "simple wear" - consider adding more variety

3. **Small dataset**: 48 images is minimal. For better results, consider:
   - Adding 20-50 more images
   - Varying lighting conditions
   - Varying backgrounds
   - Adding different poses/angles

## Ready to Train? ✅

Your dataset is **ready** for training. Choose your training script:

```bash
# Quick proof of concept (~13 min)
./train_chani_minimal.sh

# Or balanced quality (~40 min)
./train_chani_full.sh  # (but adjust epochs to 6 for balanced)
```

## Training Commands Available

1. `train_chani_minimal.sh` - Minimal compute (2 epochs, rank 16)
2. `train_chani_full.sh` - Full quality (10 epochs, rank 32)
3. Generate new balanced script with 6 epochs + rank 32




