#!/bin/bash
# Helper script to apply ROCm stability kernel parameters
# This requires sudo access and a reboot to take effect

echo "=========================================="
echo "ROCm Kernel Parameters Setup"
echo "=========================================="
echo ""
echo "This script will add the following kernel parameters to GRUB:"
echo "  - amdgpu.noretry=0    (Allow retries on GPU errors)"
echo "  - amdgpu.gpu_recovery=1 (Enable GPU recovery mechanisms)"
echo "  - amdgpu.isolation=0   (Disable GPU isolation for better memory sharing)"
echo ""
echo "These parameters help fix:"
echo "  - 'VRAM is lost due to GPU reset!' errors"
echo "  - Memory access faults"
echo "  - GPU hangs on RDNA3.5 (Strix Halo)"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Backup current GRUB config
echo ""
echo "Backing up current GRUB config..."
sudo cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Read current GRUB_CMDLINE_LINUX_DEFAULT
CURRENT_CMDLINE=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub | cut -d'=' -f2- | tr -d '"')

# Check if parameters already exist
if echo "$CURRENT_CMDLINE" | grep -q "amdgpu.noretry"; then
    echo ""
    echo "⚠️  Warning: amdgpu kernel parameters already found in GRUB_CMDLINE_LINUX_DEFAULT"
    echo "Current line: $CURRENT_CMDLINE"
    echo ""
    read -p "Replace existing parameters? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. No changes made."
        exit 1
    fi
    # Remove existing amdgpu parameters
    NEW_CMDLINE=$(echo "$CURRENT_CMDLINE" | sed -E 's/\s*amdgpu\.(noretry|gpu_recovery|isolation)=[0-9]//g')
else
    NEW_CMDLINE="$CURRENT_CMDLINE"
fi

# Add new parameters
NEW_CMDLINE="$NEW_CMDLINE amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.isolation=0"

# Update GRUB config
echo ""
echo "Updating GRUB configuration..."
sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"$NEW_CMDLINE\"|" /etc/default/grub

echo "✅ GRUB configuration updated"
echo ""
echo "New GRUB_CMDLINE_LINUX_DEFAULT:"
grep "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub
echo ""

# Update GRUB
echo "Updating GRUB bootloader..."
sudo update-grub

echo ""
echo "=========================================="
echo "✅ Kernel parameters configured!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: You must REBOOT for these changes to take effect."
echo ""
echo "After reboot, verify with:"
echo "  cat /proc/cmdline | grep amdgpu"
echo ""
echo "You should see: amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.isolation=0"
echo ""





