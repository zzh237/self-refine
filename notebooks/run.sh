#!/bin/bash
# notebooks/run.sh
# This script orchestrates the entire analysis pipeline.

set -e

# --- Setup Paths ---
NOTEBOOKS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$NOTEBOOKS_DIR/.."
ANALYSIS_DIR="$PROJECT_ROOT/notebooks"

# --- Setup Logging ---
LOG_DIR="$NOTEBOOKS_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/analysis_run_${TIMESTAMP}.log"

# This command redirects all subsequent output (stdout and stderr)
# to both the console and the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- ANALYSIS PIPELINE STARTED ---"
echo "Log file for this run: $LOG_FILE"
echo "---------------------------------"

# --- Step 1: Aggregate Raw Results ---
echo ""
echo ">>> STEP 1: Aggregating raw experiment results..."
python3 "$ANALYSIS_DIR/create_master_summary.py"
echo ">>> STEP 1 COMPLETE: master_results.csv created."

# --- Step 2: Calculate Difficulty Scores ---
echo ""
echo ">>> STEP 2: Calculating difficulty scores and bins..."
python3 "$ANALYSIS_DIR/calculate_difficulty.py"
echo ">>> STEP 2 COMPLETE: master_results_with_difficulty.csv created."

echo ""
echo "--- ANALYSIS PIPELINE FINISHED SUCCESSFULLY ---"