#!/bin/bash
# scripts/run2.sh
# This is the launcher script. Run this to start all experiments.

# --- Path Setup ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_SCRIPT_PATH="$SCRIPT_DIR/run.sh"

# --- Log File Setup ---
# Create a new directory for the main suite logs
SUITE_LOG_DIR="$SCRIPT_DIR/../logs/suite_logs"
mkdir -p "$SUITE_LOG_DIR"

# Create a timestamped log file name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_FILE="$SUITE_LOG_DIR/run_suite_${TIMESTAMP}.log"

echo "--- Starting Experiment Suite ---"

# --- Execute the Main Script in the Background ---
nohup bash "$MAIN_SCRIPT_PATH" > "$MASTER_LOG_FILE" 2>&1 &

PID=$!
echo "✅ Experiment suite launched in the background."
echo "   Process ID (PID): $PID"
echo "   Main log file is located at: $MASTER_LOG_FILE"
echo "   You can monitor the overall progress with:"
echo "   tail -f $MASTER_LOG_FILE"