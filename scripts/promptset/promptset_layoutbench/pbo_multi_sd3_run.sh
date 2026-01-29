#!/bin/bash

gpu1=0
gpu2=1
gpu3=1
objectives="sd3"
mode="low-image"
algos="qEI"   #EUBO+qEI+rand
result_path="./outputs/promptset/promptset_layoutbench/MultiBO/$mode/$objectives/"
category1="combo"
category2="size"
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_layoutbench/pbo_multi_sd3.sh $gpu1 "$objectives" "$mode" "$algos" "$result_path" "$category1" > logs/test-$mode-$objectives-$category1-multi-logit-$algos-$gpu1.log 2>&1 &
PID1=$!
# bash ./scripts/promptset/promptset_layoutbench/pbo_multi_sd3.sh $gpu2 "$objectives" "$mode" "$algos" "$result_path" "$category2" > logs/test-$mode-$objectives-$category2-multi-logit-$algos-$gpu2.log 2>&1 &
# PID2=$!


echo "Started processes:"
echo "  $objectives-$category1 on GPU $gpu1 : PID $PID1"
# echo "  $objectives-$category2 on GPU $gpu2 : PID $PID2"

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait $PID1 && echo "✓ $objectives-$category1 completed"
# wait $PID2 && echo "✓ $objectives-$category2 completed"

echo "All tasks completed!"