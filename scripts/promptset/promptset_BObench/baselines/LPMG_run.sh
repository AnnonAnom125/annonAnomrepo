#!/bin/bash

gpu1=0
gpu2=1
gpu3=1
objectives="sdxl"
mode="low-image"
result_path="./outputs/promptset/promptset_BObench/LPMG/$mode/$objectives/"
category1="part1"
category2="part2"
category3="part3"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_BObench/baselines/LPMG.sh $gpu1 "$objectives" "$mode"  "$result_path" "$category1" > logs/test-$mode-$objectives-$category1-LPMG-$gpu1.log 2>&1 &
PID1=$!
# bash ./scripts/promptset/promptset_BObench/baselines/LPMG.sh $gpu2 "$objectives" "$mode"  "$result_path" "$category2" > logs/test-$mode-$objectives-$category2-LPMG-$gpu2.log 2>&1 &
# PID2=$!
# bash ./scripts/promptset/promptset_BObench/baselines/LPMG.sh $gpu3 "$objectives" "$mode"  "$result_path" "$category3" > logs/test-$mode-$objectives-$category3-LPMG-$gpu3.log 2>&1 &
# PID3=$!

echo "Started processes:"
echo "  $objectives-$category1 on GPU $gpu1 : PID $PID1"
# echo "  $objectives-$category2 on GPU $gpu2 : PID $PID2"
# echo "  $objectives $category3 on GPU $gpu3 : PID $PID3"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID1 && echo "✓ $objectives-$category1 completed"
# wait $PID2 && echo "✓ $objectives-$category2 completed"
# wait $PID3 && echo "✓ $objectives $category3 completed"

echo "All tasks completed!"