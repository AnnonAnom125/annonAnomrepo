#!/bin/bash

gpu1=0
gpu2=2
gpu3=3
objectives="sdxl"
mode="low-image"
algos="qEI"   #EUBO+qEI+rand
category1="part7"
category2="part2"
category3="part3"

metrics="lpips-clip-ssim"  #clip, aesthetic, hpsv2, picscore, imagereward, lpips-clip-ssim


# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting parallel execution..."

bash ./scripts/promptset/promptset_BObench/pbo_multi_sdxl_non-human.sh $gpu1 "$objectives" "$mode" "$algos" "$category1" "$metrics" > logs/test-$mode-$objectives-$category1-non-human-multi-logit-$algos-$gpu1.log 2>&1 &
PID1=$!
# bash ./scripts/promptset/promptset_BObench/pbo_multi_sdxl_non-human.sh $gpu2 "$objectives" "$mode" "$algos" "$category2" "$metrics" > logs/test-$mode-$objectives-$category2-non-human-multi-logit-$algos-$gpu2.log 2>&1 &
# PID2=$!
# bash ./scripts/promptset/promptset_BObench/pbo_multi_sdxl_non-human.sh $gpu3 "$objectives" "$mode" "$algos" "$category3" "$metrics" > logs/test-$mode-$objectives-$category3-non-human-multi-logit-$algos-$gpu3.log 2>&1 &
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