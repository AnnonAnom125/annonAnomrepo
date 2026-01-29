#!/bin/sh

# Accept arguments
CUDA_VISIBLE_DEVICES="$1"
OBJECTIVES="$2"
MODE="$3" 
RESULT_PATH="$4"
CATEGORY="$5"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES


USE_RAND_SEED="True"


for category in $CATEGORY; do
    PROMPTSET_PATH="promptsets/BObench/${category}.txt"
    echo "Running job for $OBJECTIVES-$PROMPTSET_PATH"
    python -u scripts/promptset/promptset_BObench/baselines/DAS_sdxl.py \
    --output_path "$RESULT_PATH" --mode $MODE --obj_name "$OBJECTIVES" --num_samples_per_prompt 3 \
    --prompts_per_category 5 --promptset_path $PROMPTSET_PATH --use_random_seeds $USE_RAND_SEED\

done
