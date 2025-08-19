#!/bin/bash

NPROC_VALUES=(4)

OUTPUT_CSV="/workspace/chakra_fx/gemm_collected.csv"

PYTHON_SCRIPT="profile_fxgraph.py"
FIXED_ARGS="--fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama"

set -e


if [ -f "$OUTPUT_CSV" ]; then
    echo "deleting: $OUTPUT_CSV"
    rm -f "$OUTPUT_CSV"
fi

for nproc in "${NPROC_VALUES[@]}"; do
    echo "--------------------------------------------------"
    echo ">> Executing: nproc-per-node = $nproc"
    echo "--------------------------------------------------"

    torchrun --nproc-per-node="$nproc" "$PYTHON_SCRIPT" $FIXED_ARGS

    if [ $? -ne 0 ]; then
        echo "!! Error: in nproc = $nproc, exiting"
        exit 1
    fi
    echo ">> nproc = $nproc completes"
    echo ""
done

echo "=================================================="
echo "All Done"
echo "=================================================="