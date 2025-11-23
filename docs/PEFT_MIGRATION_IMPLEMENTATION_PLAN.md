# PEFT Migration Implementation Plan

## Overview

This document outlines the detailed implementation plan for migrating from the current LoRA implementation (using `musubi_tuner.networks.lora`) to Hugging Face PEFT, which has been verified to work correctly on Windows + ROCm and avoids the known tensor transfer bug.

## Current Architecture

### Current LoRA Implementation
- **Module**: `musubi_tuner.networks.lora`
- **Network Creation**: `network_module.create_arch_network()` or `network_module.create_network()`
- **Application**: `network.apply_to(transformer, apply_unet=True)`
- **Training**: Network is wrapped and trained with Accelerate
- **Device Placement**: Manual CPU→GPU transfers (affected by ROCm bug)

### Key Code Locations
1. **Network Creation** (`hv_train_network.py` lines 1800-1863):
   - Imports `musubi_tuner.networks.lora as lora_module`
   - Creates network with `create_arch_network()` or `create_network()`
   - Applies to transformer with `network.apply_to()`

2. **Training Loop** (`hv_train_network.py` lines 2550-2730):
   - Manual tensor transfers with `move_to_gpu()` function
   - Multiple fallback methods (all affected by ROCm bug)
   - Network forward pass through `call_dit()`

3. **Network Module** (`networks/lora.py`):
   - Custom LoRA implementation
   - Handles weight application and training

## Migration Strategy

### Phase 1: Create PEFT Adapter Layer (Non-Breaking)

**Goal**: Create a compatibility layer that allows PEFT to work alongside existing code.

**Steps**:
1. Create `src/musubi_tuner/networks/peft_lora.py`:
   - Wrapper that converts PEFT LoRA to match current network interface
   - Implements `create_arch_network()`, `apply_to()`, etc.
   - Maintains backward compatibility with existing code

2. Add PEFT configuration:
   - Map current LoRA parameters (`network_dim`, `network_alpha`, `network_dropout`) to PEFT `LoraConfig`
   - Handle target modules for WAN transformer

3. Test compatibility layer:
   - Verify PEFT LoRA can be created with same parameters
   - Test that it works with existing training code

### Phase 2: Integrate PEFT into Training Loop

**Goal**: Replace manual tensor transfers with PEFT's device handling.

**Steps**:
1. Modify network creation (`hv_train_network.py`):
   - Add flag to use PEFT: `--use_peft` (default: False for backward compatibility)
   - When enabled, use PEFT instead of custom LoRA
   - Keep existing code path for non-PEFT training

2. Update device placement:
   - Remove `move_to_gpu()` workarounds when using PEFT
   - Let PEFT handle device placement (it works correctly on ROCm)
   - Keep workarounds for non-PEFT path

3. Update training loop:
   - When using PEFT, use standard `.to(device)` transfers (PEFT avoids the bug)
   - Simplify tensor transfer logic for PEFT path
   - Keep existing logic for non-PEFT path

### Phase 3: Full PEFT Integration

**Goal**: Make PEFT the default and optimize for it.

**Steps**:
1. Make PEFT default:
   - Set `--use_peft` default to `True`
   - Update documentation
   - Keep old path as fallback

2. Optimize for PEFT:
   - Remove unnecessary workarounds when using PEFT
   - Simplify code paths
   - Add PEFT-specific optimizations

3. Testing and validation:
   - Full training run with PEFT
   - Verify no zero loss
   - Compare training results with old implementation

## Implementation Details

### 1. PEFT LoRA Configuration

**Current Parameters**:
- `network_dim` (r in PEFT)
- `network_alpha` (lora_alpha in PEFT)
- `network_dropout` (lora_dropout in PEFT)
- `network_module` (target modules)

**PEFT Mapping**:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=args.network_dim,
    lora_alpha=args.network_alpha,
    target_modules=target_modules,  # Need to identify WAN transformer modules
    lora_dropout=args.network_dropout or 0.0,
    bias="none",
    task_type=None,  # Not classification
)
```

### 2. Target Modules for WAN Transformer

**Challenge**: Need to identify which modules in `HYVideoDiffusionTransformer` should have LoRA applied.

**Approach**:
1. Inspect transformer architecture
2. Identify attention and feed-forward layers
3. Map to PEFT target module patterns
4. Test with different target module configurations

**Potential Target Modules**:
- Attention layers: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- Feed-forward: `fc1`, `fc2`, `gate_proj`, `up_proj`, `down_proj`
- May need to inspect actual WAN transformer structure

### 3. Network Interface Compatibility

**Current Interface**:
```python
network = network_module.create_arch_network(
    1.0,  # multiplier
    args.network_dim,
    args.network_alpha,
    vae,
    None,
    transformer,
    neuron_dropout=args.network_dropout,
    **net_kwargs,
)
network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
```

**PEFT Interface**:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(...)
peft_model = get_peft_model(transformer, lora_config)
```

**Compatibility Layer**:
- Create wrapper class that implements same interface
- Internally uses PEFT
- Maintains compatibility with existing code

### 4. Training Loop Changes

**Current (with ROCm bug)**:
```python
# Manual transfer with workarounds
latents = move_to_gpu(batch["latents"], "latents")  # Multiple fallback methods
# ... training ...
```

**With PEFT (no bug)**:
```python
# Simple transfer (PEFT avoids the bug)
latents = batch["latents"].to(accelerator.device)
# ... training ...
```

**Implementation**:
- Add conditional: `if use_peft: simple_transfer() else: move_to_gpu()`
- Keep both paths for backward compatibility

## File Changes

### New Files
1. `src/musubi_tuner/networks/peft_lora.py`:
   - PEFT LoRA wrapper
   - Compatibility layer for existing interface

2. `tests/test_peft_integration.py`:
   - Integration tests for PEFT with WAN model
   - Verify tensor transfers work
   - Test training loop

### Modified Files
1. `src/musubi_tuner/hv_train_network.py`:
   - Add `--use_peft` argument
   - Modify network creation to support PEFT
   - Update training loop for PEFT path
   - Keep existing code for backward compatibility

2. `docs/TORCHTUNE_PEFT_MIGRATION_PLAN.md`:
   - Update with implementation progress
   - Document any issues found

## Testing Plan

### Unit Tests
1. **PEFT LoRA Creation**:
   - Test `peft_lora.create_arch_network()` with various parameters
   - Verify PEFT config matches input parameters
   - Test target module identification

2. **Device Placement**:
   - Test tensor transfers with PEFT
   - Verify no zeros (ROCm bug check)
   - Test with actual WAN model

### Integration Tests
1. **Full Training Loop**:
   - Run 10 training steps with PEFT
   - Verify no zero loss
   - Check tensor integrity at each step
   - Compare with old implementation

2. **Backward Compatibility**:
   - Test that `--use_peft=False` still works
   - Verify old code path unchanged

### Validation
1. **Training Results**:
   - Run full training with PEFT
   - Verify loss decreases correctly
   - Check saved LoRA weights

2. **Performance**:
   - Compare training speed (PEFT vs old)
   - Check memory usage
   - Verify no regressions

## Risk Mitigation

### Risks
1. **Target Module Identification**: May need to inspect WAN transformer structure
   - **Mitigation**: Create test script to identify modules, iterate on target patterns

2. **Interface Compatibility**: PEFT interface may differ significantly
   - **Mitigation**: Create comprehensive compatibility layer, extensive testing

3. **Training Behavior**: PEFT may behave differently than custom LoRA
   - **Mitigation**: Compare training results, adjust parameters if needed

4. **Backward Compatibility**: Changes may break existing workflows
   - **Mitigation**: Keep old path as default initially, make PEFT opt-in

## Success Criteria

1. ✅ PEFT LoRA can be created with same parameters as current implementation
2. ✅ Training loop works with PEFT (no zero loss)
3. ✅ Tensor transfers work correctly (no ROCm bug)
4. ✅ Training results are comparable to old implementation
5. ✅ Backward compatibility maintained (old path still works)
6. ✅ Documentation updated

## Timeline

- **Phase 1** (Compatibility Layer): 2-3 days
- **Phase 2** (Integration): 2-3 days
- **Phase 3** (Optimization): 1-2 days
- **Testing and Validation**: 2-3 days

**Total**: ~1-2 weeks

## Next Steps

1. **Immediate**: Create PEFT compatibility layer (`peft_lora.py`)
2. **Next**: Identify target modules for WAN transformer
3. **Then**: Integrate into training loop
4. **Finally**: Test and validate




