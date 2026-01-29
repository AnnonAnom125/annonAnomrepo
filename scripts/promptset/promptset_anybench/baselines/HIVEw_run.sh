#!/bin/bash

gpu1=0
gpu2=1
objectives="sdxl"
mode="low-image"
result_path="./outputs/promptset/promptset_anybench/HIVEw/$mode/$objectives/"
category1="movement"
category2="resize"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_anybench/baselines/HIVEw.sh $gpu1 "$objectives" "$mode"  "$result_path" "$category1" > logs/test-$mode-$objectives-$category1-HIVEw-$gpu1.log 2>&1 &
PID1=$!
# bash ./scripts/promptset/promptset_anybench/baselines/HIVEw.sh $gpu2 "$objectives" "$mode"  "$result_path" "$category2" > logs/test-$mode-$objectives-$category2-HIVEw-$gpu2.log 2>&1 &
# PID2=$!

echo "Started processes:"
echo "  $objectives-$category1 on GPU $gpu1 : PID $PID1"
# echo "  $objectives-$category2 on GPU $gpu2 : PID $PID2"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID1 && echo "✓ $objectives-$category1 completed"
# wait $PID2 && echo "✓ $objectives-$category2 completed"

echo "All tasks completed!"