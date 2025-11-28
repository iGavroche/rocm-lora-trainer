#!/bin/bash
# Systematic test script to find the best attention mechanism for ROCm
# Tests each attention mechanism with appropriate flags

source .venv/bin/activate

# Configuration
DATASET_CONFIG="dataset.toml"
OUTPUT_DIR="outputs"
OUTPUT_NAME="chani_test_attn"
MAX_EPOCHS=1  # Just 1 epoch for testing
LEARNING_RATE=2e-4
NETWORK_DIM=16
NETWORK_ALPHA=12
GRADIENT_ACCUMULATION_STEPS=4
MIXED_PRECISION="fp16"

# Model paths
VAE_PATH="models/wan/wan_2.1_vae.safetensors"
T5_PATH="models/wan/umt5-xxl-enc-bf16.safetensors"
DIT_LOW_NOISE="models/wan/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# Test configurations: (attention_flag, split_attn_needed, description)
declare -a TEST_CONFIGS=(
    "sdpa false 'PyTorch SDPA (default, may have warnings)'"
    "xformers true 'Xformers with split_attn (recommended for ROCm)'"
    "xformers false 'Xformers without split_attn (may not work)'"
    "flash_attn false 'Flash Attention 2 (if installed)'"
    "flash_attn true 'Flash Attention 2 with split_attn'"
    "flash3 false 'Flash Attention 3 (if installed, no split_attn)'"
    "sage_attn false 'Sage Attention (if installed, no split_attn)'"
)

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/attention_tests"

echo "=========================================="
echo "Testing Attention Mechanisms for ROCm"
echo "=========================================="
echo ""
echo "This will test each attention mechanism with 1 epoch each."
echo "Check the logs in outputs/attention_tests/ for results."
echo ""
echo "Press Ctrl+C to stop after a successful test."
echo ""

for config in "${TEST_CONFIGS[@]}"; do
    read -r attn_flag split_needed description <<< "$config"
    
    test_name="${OUTPUT_NAME}_${attn_flag}"
    if [ "$split_needed" = "true" ]; then
        test_name="${test_name}_split"
    fi
    
    log_file="$OUTPUT_DIR/attention_tests/${test_name}.log"
    
    echo "=========================================="
    echo "Testing: $description"
    echo "  Flag: --$attn_flag"
    [ "$split_needed" = "true" ] && echo "  With: --split_attn"
    echo "  Log: $log_file"
    echo "=========================================="
    echo ""
    
    # Build command
    cmd="accelerate launch --num_cpu_threads_per_process 1 --mixed_precision $MIXED_PRECISION \
        src/musubi_tuner/wan_train_network.py \
        --task i2v-A14B \
        --dit $DIT_LOW_NOISE \
        --vae $VAE_PATH \
        --t5 $T5_PATH \
        --dataset_config $DATASET_CONFIG \
        --output_dir $OUTPUT_DIR \
        --output_name $test_name \
        --mixed_precision $MIXED_PRECISION \
        --$attn_flag"
    
    if [ "$split_needed" = "true" ]; then
        cmd="$cmd --split_attn"
    fi
    
    cmd="$cmd --gradient_checkpointing \
        --optimizer_type adamw \
        --learning_rate $LEARNING_RATE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --network_module networks.lora_wan \
        --network_dim $NETWORK_DIM \
        --network_alpha $NETWORK_ALPHA \
        --timestep_sampling shift \
        --discrete_flow_shift 5.0 \
        --min_timestep 0 \
        --max_timestep 900 \
        --preserve_distribution_shape \
        --max_train_epochs $MAX_EPOCHS \
        --save_every_n_epochs 1 \
        --seed 42"
    
    # Run test and capture output
    echo "Starting test..."
    if timeout 1800 bash -c "$cmd" 2>&1 | tee "$log_file"; then
        # Check if training completed successfully (look for "saving checkpoint" or similar)
        if grep -q "saving checkpoint\|Training complete\|epoch.*complete" "$log_file" 2>/dev/null; then
            echo ""
            echo "✅ SUCCESS: $description completed without crashes!"
            echo "   This configuration appears to work."
            echo ""
            read -p "Continue testing other mechanisms? (y/n): " continue_test
            if [ "$continue_test" != "y" ]; then
                echo "Stopping tests. Use this configuration:"
                echo "  --$attn_flag"
                [ "$split_needed" = "true" ] && echo "  --split_attn"
                exit 0
            fi
        else
            echo ""
            echo "⚠️  Test completed but may have had issues. Check log: $log_file"
            echo ""
        fi
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo ""
            echo "⏱️  Test timed out after 30 minutes"
        else
            echo ""
            echo "❌ FAILED: $description crashed or errored"
            echo "   Check log: $log_file"
        fi
        echo ""
        read -p "Continue to next test? (y/n): " continue_test
        if [ "$continue_test" != "y" ]; then
            exit 1
        fi
    fi
    
    echo ""
    echo "Waiting 5 seconds before next test..."
    sleep 5
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "Check logs in: $OUTPUT_DIR/attention_tests/"
echo "=========================================="




