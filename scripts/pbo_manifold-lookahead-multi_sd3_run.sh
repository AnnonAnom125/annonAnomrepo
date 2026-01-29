#!/bin/bash

gpu=1
objectives="sd3"
mode="low-image"
algos="manifold-lookahead"
result_path="./outputs/MultiBO/$mode/$objectives/"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/pbo_manifold-lookahead-multi_sd3.sh $gpu "$objectives" "$mode" "$algos" "$result_path" > logs/test-$mode-$objectives-pair-logit-$algos-$gpu-new-seed_0.log 2>&1 &
PID=$!

echo "Started processes:"
# echo "  animals on GPU $gpu1 : PID $PID1"
echo "  $objectives on GPU $gpu : PID $PID"
# echo "  objects on GPU $gpu3 : PID $PID3"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID && echo "✓ $objectives completed"

echo "All tasks completed!"