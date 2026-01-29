#!/bin/sh

# Accept arguments
CUDA_VISIBLE_DEVICES="$1"
OBJECTIVES="$2"
MODE="$3"
ALGOS="$4" 
CATEGORY="$5"
METRICS="$6"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

Q=4
T=4
LOGIT="True"
MULTI="True"
NUM_FANTASIES=2
NUM_REF_POINTS=6
USE_RAND_SEED="True"
RES_BLK="n"

HH="True"  # non-human scoring

for category in $CATEGORY; do
for metric in $METRICS; do
        RESULT_PATH="./outputs/promptset/promptset_BObench/MultiBO-non-human_${metric}/$MODE/$OBJECTIVES/"
        PROMPTSET_PATH="promptsets/BObench/${category}.txt"
        echo "Running job for $OBJECTIVES-$PROMPTSET_PATH"
        echo "Non-human scoring metric $metric"
        python -u scripts/promptset/promptset_BObench/pbo_manifold_sdxl_nonh.py \
        --acf "$ALGOS" --output_path "$RESULT_PATH" --q $Q --T "$T" --mode $MODE --logit $LOGIT \
        --multi $MULTI --obj_name "$OBJECTIVES" --num_samples_per_prompt 3 --res_blk $RES_BLK \
        --prompts_per_category 5 --promptset_path $PROMPTSET_PATH --use_random_seeds $USE_RAND_SEED --non_human_score $HH --score_metric "$metric"\

    done
done
