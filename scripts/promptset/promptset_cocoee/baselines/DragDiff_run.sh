#!/bin/bash

gpu1=0
gpu2=1
gpu3=1
objectives="sdxl"
mode="low-image"
result_path="./outputs/promptset/promptset_cocoee/DragDiff/$mode/$objectives/"
category1="cocoee"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_cocoee/baselines/DragDiff.sh $gpu1 "$objectives" "$mode"  "$result_path" "$category1" > logs/test-$mode-$objectives-$category1-DragDiff-$gpu1.log 2>&1 &
PID1=$!

echo "Started processes:"
echo "  $objectives-$category1 on GPU $gpu1 : PID $PID1"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID1 && echo "✓ $objectives-$category1 completed"

echo "All tasks completed!"