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

# Cache T5 text encoder outputs for WAN training
python wan_cache_text_encoder_outputs.py `
  --dataset_config dataset.toml `
  --t5 .\models\wan\umt5-xxl-enc-bf16.safetensors `
  --batch_size 1 `
  --num_workers 0 `
  --skip_existing

