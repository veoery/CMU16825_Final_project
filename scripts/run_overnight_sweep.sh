#!/bin/bash
# Automated overnight wandb sweep with error handling
# Run with: bash scripts/run_overnight_sweep.sh

set -e  # Exit on error initially
trap 'echo "Script interrupted at $(date)"; exit 130' INT TERM

# Configuration
SWEEP_CONFIG="sweep_overnight_5070ti.yaml"
LOG_DIR="./sweep_logs_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$LOG_DIR/sweep_summary.txt"
ERROR_LOG="$LOG_DIR/errors.log"
MAX_RUNTIME_HOURS=8
MAX_CONSECUTIVE_FAILURES=3

# Create log directory
mkdir -p "$LOG_DIR"

echo "====================================="
echo "Starting Overnight Wandb Sweep"
echo "====================================="
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo "Max runtime: $MAX_RUNTIME_HOURS hours"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p3d_5070ti

# Verify GPU is available
echo "Checking GPU availability..."
nvidia-smi > "$LOG_DIR/gpu_status.txt" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: GPU not available!" | tee -a "$ERROR_LOG"
    exit 1
fi
echo "GPU check passed"
echo ""

# Initialize wandb sweep
echo "Initializing wandb sweep..."
SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" 2>&1 | grep "Run sweep agent with:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create sweep" | tee -a "$ERROR_LOG"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID" | tee "$LOG_DIR/sweep_id.txt"
echo ""

# Start timing
START_TIME=$(date +%s)
END_TIME=$((START_TIME + MAX_RUNTIME_HOURS * 3600))

# Run counter and failure tracking
RUN_COUNT=0
CONSECUTIVE_FAILURES=0
TOTAL_FAILURES=0
SUCCESSFUL_RUNS=0

echo "Starting sweep agent..."
echo "Will run until $(date -d @$END_TIME) or max runs reached"
echo ""

# Main sweep loop with error handling
while [ $(date +%s) -lt $END_TIME ]; do
    RUN_COUNT=$((RUN_COUNT + 1))
    RUN_LOG="$LOG_DIR/run_${RUN_COUNT}_$(date +%H%M%S).log"

    echo "----------------------------------------"
    echo "Run #$RUN_COUNT at $(date)"
    echo "Successful: $SUCCESSFUL_RUNS | Failed: $TOTAL_FAILURES"
    echo "Log: $RUN_LOG"

    # Run sweep agent with timeout and error capture
    set +e  # Don't exit on error
    timeout 45m wandb agent "$SWEEP_ID" --count 1 > "$RUN_LOG" 2>&1
    EXIT_CODE=$?
    set -e

    # Analyze exit status
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run #$RUN_COUNT completed successfully"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        CONSECUTIVE_FAILURES=0

    elif [ $EXIT_CODE -eq 124 ]; then
        # Timeout (likely hanging)
        echo "✗ Run #$RUN_COUNT timed out (45 min)" | tee -a "$ERROR_LOG"
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))

        # Kill any hung Python processes
        pkill -9 -f train_curriculum.py || true

    else
        # Other error (OOM, crash, etc.)
        ERROR_TYPE="Unknown"

        # Check for OOM
        if grep -q "CUDA out of memory" "$RUN_LOG"; then
            ERROR_TYPE="OOM"
        elif grep -q "RuntimeError" "$RUN_LOG"; then
            ERROR_TYPE="Runtime Error"
        elif grep -q "KeyboardInterrupt" "$RUN_LOG"; then
            ERROR_TYPE="Interrupted"
            break
        fi

        echo "✗ Run #$RUN_COUNT failed: $ERROR_TYPE (exit code: $EXIT_CODE)" | tee -a "$ERROR_LOG"
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))

        # Extract config that failed for debugging
        echo "Failed config:" >> "$ERROR_LOG"
        grep -A 20 "config:" "$RUN_LOG" >> "$ERROR_LOG" 2>/dev/null || echo "Config not found" >> "$ERROR_LOG"
        echo "" >> "$ERROR_LOG"
    fi

    # Safety check: too many consecutive failures
    if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
        echo ""
        echo "⚠ WARNING: $MAX_CONSECUTIVE_FAILURES consecutive failures detected!"
        echo "This may indicate a systemic issue. Check $ERROR_LOG"
        echo "Continuing anyway..."
        echo ""
        CONSECUTIVE_FAILURES=0  # Reset to continue
    fi

    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Brief cooldown
    sleep 5
done

# Sweep completed
echo ""
echo "====================================="
echo "Sweep Completed!"
echo "====================================="
echo "End time: $(date)"
echo ""

# Generate summary
echo "Generating summary report..."
cat > "$SUMMARY_FILE" << EOF
===================================
OVERNIGHT SWEEP SUMMARY
===================================

Sweep ID: $SWEEP_ID
Start Time: $(date -d @$START_TIME)
End Time: $(date)
Duration: $(( ($(date +%s) - START_TIME) / 3600 ))h $(( (($(date +%s) - START_TIME) % 3600) / 60 ))m

RESULTS:
--------
Total Runs: $RUN_COUNT
Successful: $SUCCESSFUL_RUNS
Failed: $TOTAL_FAILURES
Success Rate: $(( SUCCESSFUL_RUNS * 100 / RUN_COUNT ))%

LOG LOCATIONS:
--------------
Summary: $SUMMARY_FILE
Errors: $ERROR_LOG
Individual Runs: $LOG_DIR/run_*.log
GPU Status: $LOG_DIR/gpu_status.txt

NEXT STEPS:
-----------
1. Analyze results with: python scripts/analyze_sweep_results.py --sweep_id $SWEEP_ID
2. View on wandb: https://wandb.ai/your-entity/CAD-MLLM-Hyperparam-Sweep/sweeps/$SWEEP_ID
3. Check errors: cat $ERROR_LOG

EOF

# Display summary
cat "$SUMMARY_FILE"

# Open wandb sweep page (optional - comment out if not needed)
echo ""
echo "Opening wandb sweep page..."
python -m webbrowser -t "https://wandb.ai" 2>/dev/null || echo "Visit wandb.ai to view results"

echo ""
echo "✓ Sweep complete! Check $SUMMARY_FILE for details."
