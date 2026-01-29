#!/bin/bash

gpu1=0
gpu2=1
gpu3=1
objectives="pixart"
mode="low-image"
algos="manifold-dbs"   #EUBO+qEI+rand
result_path="./outputs/promptset/promptset_cocoee/MultiBO/$mode/$objectives/"
category1="cocoee"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_cocoee/pbo_manifold-dbs-multi_pixart.sh $gpu1 "$objectives" "$mode" "$algos" "$result_path" "$category1" > logs/test-$mode-$objectives-$category1-pair-logit-$algos-$gpu1.log 2>&1 &
PID1=$!

echo "Started processes:"
echo "  $objectives-$category1 on GPU $gpu1 : PID $PID1"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID1 && echo "✓ $objectives-$category1 completed"

echo "All tasks completed!"