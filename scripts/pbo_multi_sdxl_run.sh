#!/bin/bash

gpu=0
objectives="sdxl"
mode="low-image"
algos="qEI"
result_path="./outputs/new-2026/MultiBO/$mode/$objectives/"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/pbo_multi_sdxl.sh $gpu $grad "$objectives" "$mode" "$algos" "$result_path" > logs/test-$mode-$objectives-pair-logit-$algos-$gpu-new-seed_0.log 2>&1 &
PID=$!

echo "Started processes:"
# echo "  animals on GPU $gpu1 : PID $PID1"
echo "  $objectives on GPU $gpu : PID $PID"
# echo "  objects on GPU $gpu3 : PID $PID3"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID && echo "✓ $objectives completed"

echo "All tasks completed!"