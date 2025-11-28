#!/bin/bash
# List available training checkpoints for resuming

OUTPUT_DIR="outputs"

echo "Available training checkpoints:"
echo "================================"
echo ""

# Find state directories (for resuming)
echo "State directories (for --resume):"
if ls -d "$OUTPUT_DIR"/*-state* "$OUTPUT_DIR"/*-*-state 2>/dev/null | grep -q .; then
    ls -d "$OUTPUT_DIR"/*-state* "$OUTPUT_DIR"/*-*-state 2>/dev/null | while read -r dir; do
        echo "  $dir"
    done
else
    echo "  (none found)"
fi

echo ""
echo "LoRA model files:"
if ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | grep -q .; then
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "  (none found)"
fi

echo ""
echo "To resume training, set RESUME_PATH in train_chani_full.sh:"
echo "  RESUME_PATH=\"outputs/chani_full-000001-state\""
echo ""
echo "Or use the latest checkpoint:"
LATEST_STATE=$(ls -td "$OUTPUT_DIR"/*-state* "$OUTPUT_DIR"/*-*-state 2>/dev/null | head -1)
if [ -n "$LATEST_STATE" ]; then
    echo "  RESUME_PATH=\"$LATEST_STATE\""
else
    echo "  (no state directories found)"
fi





