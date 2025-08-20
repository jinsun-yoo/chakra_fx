#!/bin/bash

# Exit immediately if any command fails
set -e

# --- Configuration Section ---
# Define the path to the CSV cache file (should match the path in your Python code)
CACHE_FILE="/workspace/chakra_fx/gemm_collected.csv"
RESULT_FILE="/workspace/chakra_fx/test_result"

# Define the base torchrun command (without --use_cache)
BASE_CMD="torchrun --nproc-per-node=4 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama"

rm -rf $RESULT_FILE
touch $RESULT_FILE

# --- Test Procedure ---

echo "==========================================================="
echo "STARTING END-TO-END PROFILING TESTS"
echo "==========================================================="

# 1. Clean up environment to ensure a fresh start
echo "[Step 0] Cleaning up previous results and cache..."
rm -f "$CACHE_FILE"
rm -rf "test_output/" # Clean up previous output directory
mkdir test_output
echo "Cleanup complete."
echo "-----------------------------------------------------------"


# 2. Test use_cache=0 (no cache, baseline test)
echo "[Step 1] Running test with use_cache=0 (Baseline)..." > $RESULT_FILE
# Use the 'time' command to measure the execution time of the entire command
$BASE_CMD --use_cache 0
echo "Test with use_cache=0 finished."
echo "-----------------------------------------------------------" >> $RESULT_FILE


# 3. Test use_cache=1 (first run, generate cache)
# This run will be slow because it needs to measure all operators and write to the CSV file
echo "[Step 2] Running test with use_cache=1 (First Run - Cache Generation)..." >> $RESULT_FILE
# Clean up again to ensure this run independently generates the cache
rm -f "$CACHE_FILE"
rm -rf "test_output/"
mkdir test_output
$BASE_CMD --use_cache 1
echo "Test with use_cache=1 (First Run) finished. Cache file should now exist."
echo "-----------------------------------------------------------" >> $RESULT_FILE


# 4. Test use_cache=1 (second run, use cache)
# This run should be very fast because it will read all durations from the CSV file
echo "[Step 3] Running test with use_cache=1 (Second Run - With Cache Hit)..." >> $RESULT_FILE
rm -rf "test_output/" # Only clean output, do not remove cache file
mkdir test_output
$BASE_CMD --use_cache 1
echo "Test with use_cache=1 (Second Run) finished."
echo "-----------------------------------------------------------" >> $RESULT_FILE


# 5. Test use_cache=2 (skip measurement)
# This run should theoretically be the fastest, as it neither measures nor reads/writes the CSV
echo "[Step 4] Running test with use_cache=2 (Skip Measurement)..." >> $RESULT_FILE
rm -rf "test_output/"
mkdir test_output
$BASE_CMD --use_cache 2
echo "Test with use_cache=2 finished."
echo "===========================================================" >> $RESULT_FILE
echo "All tests completed! "
echo "==========================================================="