#!/bin/bash
# Quick script to remove problematic kernel parameters that limit GPU memory
# This removes: amd_iommu=off, amdgpu.gttsize, and ttm.pages_limit

echo "=========================================="
echo "Removing Problematic Kernel Parameters"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - amd_iommu=off (causes memory mapping issues)"
echo "  - amdgpu.gttsize=* (may limit GPU memory incorrectly)"
echo "  - ttm.pages_limit=* (may limit GPU memory incorrectly)"
echo ""
echo "These parameters were causing GPU memory to show as 15.49 GiB instead of 96GB"
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

# Remove problematic parameters
NEW_CMDLINE=$(echo "$CURRENT_CMDLINE" | \
    sed -E 's/\s*amd_iommu=[^ ]+//g' | \
    sed -E 's/\s*amdgpu\.gttsize=[0-9]+//g' | \
    sed -E 's/\s*ttm\.pages_limit=[0-9]+//g')

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
echo "✅ Problematic parameters removed!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: You must REBOOT for these changes to take effect."
echo ""
echo "After reboot, verify GPU memory with:"
echo "  rocm-smi --showmeminfo vram"
echo "  or"
echo "  python3 -c 'import torch; print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")'"
echo ""






