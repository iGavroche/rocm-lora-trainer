#!/bin/bash
# Generate video with monitoring and crash detection

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Output files
LOG_FILE="generation_full.log"
ERROR_LOG="generation_errors.log"

# Clean previous logs
> "$LOG_FILE"
> "$ERROR_LOG"

echo "Starting video generation with monitoring..."
echo "Logging to: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo ""

# Function to check if generation is running
check_process() {
    if ! pgrep -f "wan_generate_video.py" > /dev/null; then
        return 1  # Process not running
    fi
    return 0  # Process running
}

# Run generation in background and capture all output
bash generate_chani_video.sh > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERROR_LOG" >&2) &
GENERATION_PID=$!

echo "Generation started with PID: $GENERATION_PID"
echo ""

# Monitor the process
while check_process; do
    # Show recent log entries
    tail -5 "$LOG_FILE" 2>/dev/null | tail -2 || echo "Waiting for output..."
    
    # Check for errors in error log
    if grep -q "RuntimeError\|Traceback\|Error\|Memory access fault" "$ERROR_LOG" 2>/dev/null; then
        echo ""
        echo "ERROR DETECTED!"
        echo "Last error lines:"
        tail -20 "$ERROR_LOG"
        exit 1
    fi
    
    sleep 5
done

# Wait for background process to finish and capture exit code
wait $GENERATION_PID
EXIT_CODE=$?

echo ""
echo "Generation process finished with exit code: $EXIT_CODE"

# Check if output file was created
if [ -f "outputs/chani_output.mp4" ]; then
    echo "✅ Video generated successfully: outputs/chani_output.mp4"
    ls -lh outputs/chani_output.mp4
else
    echo "❌ Video file not found!"
    echo "Last 30 lines of log:"
    tail -30 "$LOG_FILE"
    exit 1
fi

exit $EXIT_CODE





