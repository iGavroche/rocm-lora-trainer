# AMD Driver Logs Analysis

## Summary
Checked AMD Adrenalin driver logs for errors related to ROCm tensor corruption issues.

## Driver Information
- **Driver Version**: 25.20.29.01 (dated 11/6/2025)
- **Driver Date**: 2025-11-06
- **GPU**: AMD Radeon(TM) 8060S Graphics (ASIC 1586 C1 801D 2014)
- **Architecture**: Strix Halo (gfx1151)

## Log Locations Checked

### 1. AMD Install Manager Logs
- **Location**: `C:\Program Files\AMD\AMDInstallManager\Logs\`
- **Findings**: 
  - Recent logs show driver updates and checks
  - No errors related to ROCm, HIP, or GPU operations
  - Driver is up to date

### 2. Radeon Software (CNext) Logs
- **Location**: `C:\Users\Nino\AppData\Local\AMD\CN\`
- **Latest Log**: `RSX_Common.log_2025-11-22_8_13_49.log` (10.2 KB)
- **Findings**:
  - Logs contain driver update checks and UI operations
  - No ROCm/HIP errors found
  - No GPU memory errors
  - No tensor-related errors
  - Some JSON parsing failures (unrelated to GPU operations)

### 3. Windows Event Logs
- **System Log**: Checked for AMD-related events
- **Findings**:
  - `AMDRyzenMasterDriverV30` service fails to start (unrelated to GPU/ROCm)
  - No GPU driver crashes or errors
  - No ROCm-related events

### 4. Driver Files
- **Location**: `C:\Windows\System32\drivers\`
- **AMD Driver Files Found**:
  - `amdhip64.dll` (dated 3/26/2025)
  - `amdhip64_6.dll` (dated 10/27/2025)
  - `amdhip64_7.dll` (dated 10/27/2025)
  - Various other AMD driver files

## Key Findings

### ✅ No Driver-Level Errors
- No GPU driver crashes in Windows Event Logs
- No ROCm/HIP errors in Adrenalin logs
- No memory allocation errors
- No tensor-related errors

### ⚠️ Observations
1. **ROCm Logs Not in Adrenalin**: ROCm errors may not be logged to Adrenalin logs. They might be:
   - In PyTorch/ROCm runtime logs
   - In kernel-level logs (not accessible without admin tools)
   - Not logged at all (silent failures)

2. **Driver Version**: Using relatively recent driver (11/6/2025), but there may be newer versions available

3. **HIP DLLs**: Multiple HIP DLL versions present (`amdhip64.dll`, `amdhip64_6.dll`, `amdhip64_7.dll`), suggesting multiple ROCm installations

## Implications

The lack of driver-level errors suggests:

1. **The issue is likely at the PyTorch/ROCm runtime level**, not the driver level
2. **Silent failures**: The driver may be silently failing operations without logging errors
3. **Memory corruption**: The issue might be memory corruption that doesn't trigger driver errors but causes data loss

## Recommendations

1. **Check PyTorch/ROCm Runtime Logs**:
   - Look for PyTorch debug logs
   - Check if `PYTORCH_DEBUG=1` or similar environment variables are set
   - Check ROCm runtime logs (if they exist)

2. **Enable Verbose ROCm Logging**:
   - Set `HIP_LAUNCH_BLOCKING=1` to enable synchronous operations
   - Set `AMD_LOG_LEVEL=3` or similar for verbose logging
   - Check if ROCm has specific logging environment variables

3. **Check for Memory Issues**:
   - The issue might be memory fragmentation or allocation failures that don't trigger driver errors
   - Consider checking GPU memory state during training

4. **Driver Update**:
   - Check for newer driver versions specifically for Strix Halo/gfx1151
   - Consider trying a different driver version if available

5. **Kernel-Level Debugging**:
   - Use tools like `rocm-smi` or `rocminfo` to check GPU state
   - Check Windows Performance Monitor for GPU metrics during training

## Next Steps

Since driver logs don't show errors, the issue is likely:
- **PyTorch/ROCm runtime issue**: The problem is in how PyTorch interacts with ROCm
- **Memory management issue**: Memory corruption that doesn't trigger driver errors
- **Race condition**: Timing issue that only occurs in specific training scenarios

The fact that all isolated tests pass but training fails suggests a very specific interaction between:
- The training loop
- Model forward/backward passes
- Memory allocation patterns
- ROCm runtime state

This aligns with the hypothesis that the issue is in the training code's specific sequence of operations, not a general ROCm bug.


