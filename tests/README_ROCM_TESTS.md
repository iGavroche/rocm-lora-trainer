# ROCm Diagnostic Tests

These tests help isolate and identify the ROCm tensor transfer issue without running the full training script.

## Test Files

### 1. `test_rocm_tensor_transfer_issue.py`
**Purpose**: Tests basic tensor transfers in isolation

**What it tests**:
- Simple tensor transfers
- Training-size tensor transfers
- Cache file tensor transfers
- All 6 transfer methods

**Result**: ✅ All tests pass - transfers work in isolation

**Conclusion**: The issue is NOT with basic ROCm tensor transfers

### 2. `test_rocm_training_context_issue.py`
**Purpose**: Tests transfers in the training context (Accelerate + DataLoader)

**What it tests**:
- Accelerate setup
- DataLoader with collator
- Batch tensor properties
- Transfer after DataLoader

**Result**: ✅ All tests pass - transfers work in training context

**Conclusion**: The issue is NOT with Accelerate or DataLoader setup

### 3. `test_rocm_model_loading_issue.py`
**Purpose**: Tests if model loading corrupts GPU state

**What it tests**:
- Transfer before model loading
- Model loading (VAE + DiT)
- Transfer after model loading

**Result**: ✅ All tests pass - model loading doesn't corrupt state

**Conclusion**: The issue is NOT with model loading

## Key Findings

**All isolated tests pass**, but training fails. This means:

1. ✅ ROCm tensor transfers work correctly
2. ✅ Accelerate setup works correctly
3. ✅ DataLoader works correctly
4. ✅ Model loading doesn't corrupt state
5. ❌ Something in the **actual training loop** causes the issue

## Possible Causes

Since all isolated tests pass, the issue must be:

1. **Training loop sequence**: A specific sequence of operations in the loop
2. **Optimizer state**: Something about how the optimizer is set up
3. **Gradient computation**: Forward/backward pass corrupting tensors
4. **Accelerate prepare()**: How Accelerate prepares the model/optimizer
5. **Memory state**: GPU memory state during actual training
6. **Timing issue**: Race condition or synchronization issue

## Next Steps

1. Check the training loop code for differences from the tests
2. Look at how Accelerate prepares the model/optimizer
3. Check if there's something about the forward pass
4. Examine the exact sequence when transfer fails in training

## Running the Tests

```bash
# Activate environment
..\..\ComfyUI\.venv\Scripts\activate

# Run all tests
python tests/test_rocm_tensor_transfer_issue.py
python tests/test_rocm_training_context_issue.py
python tests/test_rocm_model_loading_issue.py
```

## Test Results Summary

**ALL TESTS PASS** ✅

1. ✅ `test_rocm_tensor_transfer_issue.py` - Basic transfers work
2. ✅ `test_rocm_training_context_issue.py` - Training context works
3. ✅ `test_rocm_model_loading_issue.py` - Model loading doesn't corrupt state
4. ✅ `test_rocm_accelerate_prepare_issue.py` - Accelerate prepare() works
5. ✅ `test_rocm_training_step_issue.py` - Training steps work
6. ✅ `test_rocm_exact_training_loop.py` - Exact training loop works
7. ✅ `test_rocm_with_actual_model.py` - Works even with model on GPU

## Critical Finding

**All isolated tests pass, but actual training fails.**

This means:
- ✅ ROCm tensor transfers work correctly
- ✅ Accelerate setup works correctly
- ✅ DataLoader works correctly
- ✅ Model loading doesn't corrupt state
- ✅ Training steps work correctly
- ❌ Something very specific in the **actual training run** causes the issue

## Possible Causes (Since All Tests Pass)

Since every isolated test passes, the issue must be:

1. **Timing/Race Condition**: Something about the exact timing in the real training
2. **Model-Specific**: The actual WAN model forward pass does something different
3. **Memory Fragmentation**: After many training steps, GPU memory becomes fragmented
4. **Accelerate State**: Something about how Accelerate manages state during long training runs
5. **Sequence-Specific**: A specific sequence of operations that only happens in the real training

## Next Steps

1. Check the actual training logs to see exactly when it fails
2. Look at GPU memory usage during training
3. Check if the issue occurs after a specific number of steps
4. Examine the actual WAN model forward pass code
5. Check if there's something about the training loop that's different from the tests

