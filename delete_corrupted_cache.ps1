# Delete corrupted WAN cache files
# These cache files have extreme values that cause bfloat16->float32 conversion to produce zeros

Write-Host "Deleting corrupted WAN cache files..."
Get-ChildItem -Path "myface" -Filter "*_wan.safetensors" | Remove-Item -Force
Write-Host "Deleted all WAN cache files. Now re-encode with:"
Write-Host "python src/musubi_tuner/wan_cache_latents.py --dataset_config dataset.toml --vae models/wan/wan_2.1_vae.safetensors --i2v"



