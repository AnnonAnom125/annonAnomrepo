#!/bin/sh

# Accept arguments
CUDA_VISIBLE_DEVICES="$1"
OBJECTIVES="$2"
MODE="$3"
ALGOS="$4" 
RESULT_PATH="$5"
CATEGORY="$6"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

Q=4
T=4
LOGIT="True"
MULTI="True"
NUM_FANTASIES=2
NUM_REF_POINTS=6
USE_RAND_SEED="True"


for category in $CATEGORY; do
    PROMPTSET_PATH="promptsets/LayoutBench/${category}.txt"
    echo "Running job for $OBJECTIVES-$PROMPTSET_PATH"
    python -u scripts/promptset/promptset_layoutbench/pbo_manifold_flux.py \
    --acf "$ALGOS" --output_path "$RESULT_PATH" --q $Q --T "$T" --mode $MODE --logit $LOGIT \
    --multi $MULTI --obj_name "$OBJECTIVES" --num_samples_per_prompt 3 \
    --prompts_per_category 5 --promptset_path $PROMPTSET_PATH --use_random_seeds $USE_RAND_SEED\

done
