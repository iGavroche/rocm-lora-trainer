$env:PYTHONIOENCODING="utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
# ROCm workaround: Disable IPC to avoid multiprocessing issues on Windows
$env:HIP_DISABLE_IPC="1"
# ROCm GPU architecture override for Strix Halo (gfx1151)
$env:HSA_OVERRIDE_GFX_VERSION="11.5.1"

# ROCm Configuration - Reduced logging for performance
$env:HIP_LAUNCH_BLOCKING="0"
$env:AMD_LOG_LEVEL="1"
$env:HIP_VISIBLE_DEVICES="0"
$env:ROCM_DEBUG="0"
$env:HIP_PROFILE="0"

# PyTorch Configuration - Minimal logging
$env:TORCH_LOGS=""
$env:TORCH_SHOW_CPP_STACKTRACES="0"
$env:TORCH_USE_CUDA_DSA="0"
$env:PYTORCH_CUDA_ALLOC_DEBUG="0"

# Activate ComfyUI virtual environment
$venvPath = "..\..\ComfyUI\.venv"
$venvActivate = "$venvPath\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "Activated ComfyUI virtual environment"
} else {
    Write-Host "Warning: ComfyUI venv not found at $venvPath"
    exit 1
}

# Cache latents for WAN I2V training
# Note: For I2V training, CLIP is typically required but not specified in training script
# The cache script will auto-detect I2V mode if --i2v flag is set
# If you have CLIP model, add: --clip .\models\wan\clip_model.safetensors
python wan_cache_latents.py `
  --dataset_config dataset.toml `
  --vae .\models\wan\wan_2.1_vae.safetensors `
  --i2v `
  --batch_size 1 `
  --num_workers 1 `
  --vae_dtype float32 `
  --skip_existing

