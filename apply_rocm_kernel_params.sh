#!/bin/bash
# Helper script to apply ROCm stability kernel parameters
# This requires sudo access and a reboot to take effect

echo "=========================================="
echo "ROCm Kernel Parameters Setup"
echo "=========================================="
echo ""
echo "This script will add the following kernel parameters to GRUB:"
echo "  - amdgpu.noretry=0         (Allow retries on GPU errors)"
echo "  - amdgpu.gpu_recovery=1    (Enable GPU recovery mechanisms)"
echo "  - amdgpu.isolation=0       (Disable GPU isolation for better memory sharing)"
echo "  - amdgpu.lockup_timeout=10000 (10s timeout to recover from GPU hangs)"
echo "  - amdgpu.msi=0             (Disable MSI interrupts, can fix stability issues)"
echo "  - amd_iommu=off            (Disable IOMMU for Strix Halo stability)"
echo "  - ttm.pages_limit=25165824 (TTM pages limit for 96GB system - 4KB pages)"
echo "  - ttm.page_pool_size=25165824 (TTM page pool size, should match pages_limit)"
echo ""
echo "NOTE: amdgpu.gttsize is DEPRECATED and not used (replaced by ttm.pages_limit)"
echo ""
echo "These parameters help fix:"
echo "  - 'VRAM is lost due to GPU reset!' errors"
echo "  - Memory access faults"
echo "  - GPU hangs on RDNA3.5 (Strix Halo)"
echo "  - Uninterruptible sleep (D state) hangs"
echo "  - Driver lockups during forward pass"
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
if echo "$CURRENT_CMDLINE" | grep -q "amdgpu\."; then
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
    # Remove existing amdgpu, amd_iommu, and ttm parameters
    NEW_CMDLINE=$(echo "$CURRENT_CMDLINE" | \
        sed -E 's/\s*amdgpu\.([a-z_]+)=[0-9]+//g' | \
        sed -E 's/\s*amd_iommu=[^ ]+//g' | \
        sed -E 's/\s*ttm\.pages_limit=[0-9]+//g' | \
        sed -E 's/\s*ttm\.page_pool_size=[0-9]+//g')
else
    NEW_CMDLINE="$CURRENT_CMDLINE"
fi

# Add new parameters
# Core stability parameters
NEW_CMDLINE="$NEW_CMDLINE amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.isolation=0 amdgpu.lockup_timeout=10000 amdgpu.msi=0"
# IOMMU disabled for Strix Halo stability (user confirmed this helps)
NEW_CMDLINE="$NEW_CMDLINE amd_iommu=off"
# TTM memory management for 96GB system
# Calculation: 96GB = 96 * 1024 * 1024 * 1024 / 4096 = 25,165,824 pages (4KB pages)
# Both parameters should be set to the same value
# NOTE: amdgpu.gttsize is DEPRECATED and should NOT be used
NEW_CMDLINE="$NEW_CMDLINE ttm.pages_limit=25165824 ttm.page_pool_size=25165824"

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
echo "  cat /proc/cmdline | grep -E 'amdgpu|amd_iommu|ttm'"
echo ""
echo "You should see:"
echo "  amdgpu.noretry=0 amdgpu.gpu_recovery=1 amdgpu.isolation=0 amdgpu.lockup_timeout=10000 amdgpu.msi=0"
echo "  amd_iommu=off"
echo "  ttm.pages_limit=25165824 ttm.page_pool_size=25165824"
echo ""
echo "Verify TTM settings:"
echo "  cat /sys/module/ttm/parameters/pages_limit"
echo "  cat /sys/module/ttm/parameters/page_pool_size"
echo "  (Both should show: 25165824)"
echo ""
echo "Additional helpful parameters (if issues persist):"
echo "  - amdgpu.vm_fragment_size=9 (larger VM fragments, may help with large models)"
echo "  - amdgpu.vm_update_mode=0 (synchronous VM updates, more stable)"
echo "  - amdgpu.dc=0 (disable display core, if not using GPU for display)"
echo ""



