$env:PYTHONIOENCODING="utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
# ROCm workaround: Disable IPC to avoid multiprocessing issues on Windows
$env:HIP_DISABLE_IPC="1"
# ROCm GPU architecture override for Strix Halo (gfx1151)
# This ensures ROCm uses the correct GPU architecture
$env:HSA_OVERRIDE_GFX_VERSION="11.5.1"

# ROCm Configuration
# HIP_LAUNCH_BLOCKING: Disable for better performance (set to "1" for debugging)
$env:HIP_LAUNCH_BLOCKING="0"
# AMD_LOG_LEVEL: Logging level (0=off, 1=error, 2=warn, 3=info, 4=debug)
$env:AMD_LOG_LEVEL="1"
# HIP_VISIBLE_DEVICES: Ensure we're using the correct device
$env:HIP_VISIBLE_DEVICES="0"
# ROCM_DEBUG: Additional ROCm debug flags (disable for performance)
$env:ROCM_DEBUG="0"
# HIP_PROFILE: Disable HIP profiling for performance
$env:HIP_PROFILE="0"

# PyTorch Configuration - Reduced logging for performance
# TORCH_LOGS: Minimal logging (set to "+dynamo,+inductor,+autograd,+distributed" for debugging)
$env:TORCH_LOGS=""
# TORCH_SHOW_CPP_STACKTRACES: Disable for performance
$env:TORCH_SHOW_CPP_STACKTRACES="0"
# TORCH_USE_CUDA_DSA: Disable for performance
$env:TORCH_USE_CUDA_DSA="0"
# PYTORCH_CUDA_ALLOC_DEBUG: Disable for performance
$env:PYTORCH_CUDA_ALLOC_DEBUG="0"

# Activate ComfyUI virtual environment
$venvPath = "..\..\ComfyUI\.venv"
$venvActivate = "$venvPath\Scripts\Activate.ps1"
$venvPython = "$venvPath\Scripts\python.exe"
$venvUv = "$venvPath\Scripts\uv.exe"

if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "Activated ComfyUI virtual environment"
} else {
    Write-Host "Warning: ComfyUI venv not found at $venvPath"
    exit 1
}

# Check and upgrade ROCm nightly builds for gfx1151 (Strix Halo) - only once per day
$lastCheckFile = ".lastlibcheck"
$shouldCheck = $false

if (Test-Path $lastCheckFile) {
    $lastCheckTime = (Get-Item $lastCheckFile).LastWriteTime
    $timeSinceLastCheck = (Get-Date) - $lastCheckTime
    if ($timeSinceLastCheck.TotalHours -ge 24) {
        $shouldCheck = $true
        Write-Host "Last ROCm check was $([math]::Round($timeSinceLastCheck.TotalHours, 1)) hours ago. Checking for updates..."
    } else {
        Write-Host "Last ROCm check was $([math]::Round($timeSinceLastCheck.TotalHours, 1)) hours ago. Skipping check (checks once per day)."
    }
} else {
    $shouldCheck = $true
    Write-Host "No previous ROCm check found. Checking for updates..."
}

if ($shouldCheck -and (Test-Path $venvPython)) {
    $currentVersion = & $venvPython -c "import torch; print(torch.__version__)"
    Write-Host "Current PyTorch version: $currentVersion"
    Write-Host "Upgrading to latest ROCm nightly for gfx1151..."
    & $venvUv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision torchsde --upgrade
    if ($LASTEXITCODE -eq 0) {
        $newVersion = & $venvPython -c "import torch; print(torch.__version__)"
        Write-Host "Upgraded to PyTorch version: $newVersion"
        # Update last check timestamp
        Set-Content -Path $lastCheckFile -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    } else {
        Write-Host "Warning: ROCm upgrade failed, continuing with current version"
        # Still update timestamp to avoid repeated failures
        Set-Content -Path $lastCheckFile -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }
    
    # Also upgrade Accelerate, transformers, and bitsandbytes to latest compatible versions
    Write-Host "Upgrading Accelerate, transformers, and bitsandbytes to latest compatible versions..."
    # Upgrade transformers first to ensure compatibility with huggingface-hub
    & $venvUv pip install transformers --upgrade
    if ($LASTEXITCODE -eq 0) {
        $transVersion = & $venvPython -c "import transformers; print(transformers.__version__)" 2>$null
        if ($transVersion) {
            Write-Host "Upgraded to transformers version: $transVersion"
        }
    }
    # Then upgrade Accelerate
    & $venvUv pip install accelerate --upgrade
    if ($LASTEXITCODE -eq 0) {
        $accelVersion = & $venvPython -c "import accelerate; print(accelerate.__version__)" 2>$null
        if ($accelVersion) {
            Write-Host "Upgraded to Accelerate version: $accelVersion"
        }
    } else {
        Write-Host "Warning: Accelerate upgrade failed, continuing with current version"
    }
    # Upgrade bitsandbytes to match ROCm version (important after ROCm updates)
    & $venvUv pip install bitsandbytes --upgrade
    if ($LASTEXITCODE -eq 0) {
        $bnbVersion = & $venvPython -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>$null
        if ($bnbVersion) {
            Write-Host "Upgraded to bitsandbytes version: $bnbVersion"
        }
    } else {
        Write-Host "Warning: bitsandbytes upgrade failed, continuing with current version"
    }
    # Ensure huggingface-hub is compatible (downgrade if needed)
    & $venvUv pip install "huggingface-hub<1.0" --upgrade
    if ($LASTEXITCODE -eq 0) {
        $hubVersion = & $venvPython -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>$null
        if ($hubVersion) {
            Write-Host "Ensured huggingface-hub version: $hubVersion (compatible with transformers)"
        }
    }
} elseif (-not (Test-Path $venvPython)) {
    Write-Host "Warning: Python not found in venv, skipping ROCm upgrade check"
}

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py `
  --task i2v-A14B `
  --dit .\models\wan\wan2.2_i2v_low_noise_14B_fp16.safetensors `
  --vae .\models\wan\wan_2.1_vae.safetensors `
  --t5 .\models\wan\umt5-xxl-enc-bf16.safetensors `
  --dataset_config dataset.toml `
  --network_module networks.lora_wan `
  --network_dim 32 `
  --network_alpha 32 `
  --timestep_boundary 900 `
  --timestep_sampling uniform `
  --discrete_flow_shift 5.0 `
  --preserve_distribution_shape `
  --mixed_precision fp16 `
  --sdpa `
  --optimizer_type AdamW `
  --learning_rate 1e-4 `
  --gradient_checkpointing `
  --max_train_epochs 2 `
  --save_every_n_epochs 1 `
  --output_dir ./output `
  --output_name chani_i2v_dim16_96gb `
  --max_data_loader_n_workers 0 `
