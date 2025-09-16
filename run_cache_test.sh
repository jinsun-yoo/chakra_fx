#!/bin/bash

set -e

CACHE_FILE="/workspace/chakra_fx/gemm_collected.csv"
RESULT_FILE="/workspace/chakra_fx/test_result"

BASE_CMD="torchrun --nproc-per-node=4 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama"

rm -rf $RESULT_FILE
touch $RESULT_FILE

echo "==========================================================="
echo "STARTING END-TO-END PROFILING TESTS"
echo "==========================================================="

echo "[Step 0] Cleaning up previous results and cache..."
rm -f "$CACHE_FILE"
rm -rf "test_output/"
mkdir test_output
echo "Cleanup complete."
echo "-----------------------------------------------------------"


echo "[Step 1] Running test with use_cache=0 (Baseline)..." > $RESULT_FILE

$BASE_CMD --use_cache 0
echo "Test with use_cache=0 finished."
echo "-----------------------------------------------------------" >> $RESULT_FILE

echo "[Step 2] Running test with use_cache=1 (First Run - Cache Generation)..." >> $RESULT_FILE

rm -f "$CACHE_FILE"
rm -rf "test_output/"
mkdir test_output
$BASE_CMD --use_cache 1
echo "Test with use_cache=1 (First Run) finished. Cache file should now exist."
echo "-----------------------------------------------------------" >> $RESULT_FILE

echo "[Step 3] Running test with use_cache=1 (Second Run - With Cache Hit)..." >> $RESULT_FILE
rm -rf "test_output/"
mkdir test_output
$BASE_CMD --use_cache 1
echo "Test with use_cache=1 (Second Run) finished."
echo "-----------------------------------------------------------" >> $RESULT_FILE


echo "[Step 4] Running test with use_cache=2 (Skip Measurement)..." >> $RESULT_FILE
rm -rf "test_output/"
mkdir test_output
$BASE_CMD --use_cache 2
echo "Test with use_cache=2 finished."
echo "===========================================================" >> $RESULT_FILE
echo "All tests completed! "
echo "==========================================================="