# PowerShell script to run WAN 2.2 I2V LoRA training for Chani
# This script calls the bash script train_chani_full.sh

param(
    [string]$ResumePath = ""
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "WAN 2.2 I2V LoRA Training for Chani" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're on Windows
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "Detected Windows environment" -ForegroundColor Yellow
    
    # Try to find WSL or Git Bash
    $wslPath = Get-Command wsl -ErrorAction SilentlyContinue
    $bashPath = Get-Command bash -ErrorAction SilentlyContinue
    
    if ($wslPath) {
        Write-Host "Using WSL to run bash script..." -ForegroundColor Green
        $scriptPath = "train_chani_full.sh"
        
        if ($ResumePath) {
            wsl bash -c "cd /mnt/$($PWD.Drive.Name.ToLower())/$($PWD.Path.Replace(':', '').Replace('\', '/')) && export RESUME_PATH='$ResumePath' && bash $scriptPath"
        } else {
            wsl bash -c "cd /mnt/$($PWD.Drive.Name.ToLower())/$($PWD.Path.Replace(':', '').Replace('\', '/')) && bash $scriptPath"
        }
    }
    elseif ($bashPath) {
        Write-Host "Using Git Bash to run bash script..." -ForegroundColor Green
        $scriptPath = Join-Path $PWD "train_chani_full.sh"
        
        if ($ResumePath) {
            $env:RESUME_PATH = $ResumePath
        }
        
        & bash $scriptPath
    }
    else {
        Write-Host "ERROR: Neither WSL nor Git Bash found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install one of the following:" -ForegroundColor Yellow
        Write-Host "  1. WSL (Windows Subsystem for Linux):" -ForegroundColor Yellow
        Write-Host "     wsl --install" -ForegroundColor White
        Write-Host "  2. Git Bash (comes with Git for Windows)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Alternatively, you can run the bash script directly:" -ForegroundColor Yellow
        Write-Host "  bash train_chani_full.sh" -ForegroundColor White
        exit 1
    }
}
else {
    # On Linux/Mac, just run bash directly
    Write-Host "Detected Linux/Mac environment, running bash script directly..." -ForegroundColor Green
    
    if ($ResumePath) {
        $env:RESUME_PATH = $ResumePath
    }
    
    bash train_chani_full.sh
}

Write-Host ""
Write-Host "Training script completed!" -ForegroundColor Green



