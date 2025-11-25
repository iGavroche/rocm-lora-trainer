# Attention Mechanism Testing Guide for ROCm

## Quick Reference

Based on the codebase analysis:

| Attention Mechanism | Requires `--split_attn`? | Notes |
|---------------------|-------------------------|-------|
| `--sdpa` | No | PyTorch built-in (default), may have warnings on ROCm |
| `--xformers` | **Yes** (recommended) | Should use `--split_attn` for ROCm compatibility |
| `--flash_attn` | Optional | Can work with or without `--split_attn` |
| `--flash3` | **No** | Does NOT support `--split_attn` |
| `--sage_attn` | **No** | Does NOT support `--split_attn` |

## Current Status

You're currently testing `--xformers`. According to the code:
- **You should add `--split_attn`** when using `--xformers` on ROCm
- The code shows: `assert not split_attn or cu_seqlens_q is None, "Xformers only supports splitting"`

## Testing Strategy

### Option 1: Quick Manual Test (Current Setup)

Edit `train_chani_full.sh` line 104 to test different combinations:

```bash
# Test 1: xformers with split_attn (RECOMMENDED)
--xformers \
--split_attn \

# Test 2: sdpa (default, may have warnings)
--sdpa \

# Test 3: flash_attn (if installed)
--flash_attn \

# Test 4: flash_attn with split_attn
--flash_attn \
--split_attn \
```

### Option 2: Automated Testing Script

Run the automated test script:

```bash
./test_attention_mechanisms.sh
```

This will:
- Test each attention mechanism systematically
- Run 1 epoch each to quickly identify what works
- Save logs to `outputs/attention_tests/`
- Stop when it finds a working configuration

## What to Look For

### Success Indicators:
- Training starts without errors
- No GPU permission faults in `dmesg`
- Progress bar shows steps completing
- Checkpoint saves successfully

### Failure Indicators:
- `PERMISSION_FAULTS: 0x3` in dmesg
- GPU resets
- Immediate crashes
- Import errors (library not installed)

## Recommended Testing Order

1. **`--xformers --split_attn`** ← Start here (most likely to work on ROCm)
2. **`--sdpa`** (fallback, built-in)
3. **`--flash_attn`** (if installed)
4. **`--flash_attn --split_attn`** (if flash_attn works)

## Quick Fix for Current Setup

Since you're already using `--xformers`, add `--split_attn`:

```bash
--xformers \
--split_attn \
```

This is the recommended combination for ROCm according to the codebase.






