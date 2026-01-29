#!/bin/sh

# Accept arguments
CUDA_VISIBLE_DEVICES="$1"
OBJECTIVES="$2"
MODE="$3"
ALGOS="$4" 
RESULT_PATH="$5"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

PROMPTS="The fragrant flowers bloomed on the sturdy stem and the thorny bush."
# "a couch , a table and a lamp"
#"a cat and a dog"
T_EDIT=0.0
DELTA=0.2
IMG_SEED=320923

NUM_TRIALS=1
NUM_BATCHES=15
DIM=24 #composite = 10, affine = 14, geo(a) = 48 geo(b) = 24 (grid=3) = 38 (grid=4)
NUM_INITIAL_SAMPLES=20
M=50
NUM_RESTARTS=50 #3
RAW_SAMPLES=4096
Q=4
T=4
LOGIT="True"
MULTI="True"
PLOTTING="False"
SAVE_RESULTS="True"
EDIT_TYPE="geometric"

NUM_POP_MODELS=2
NUM_POP_DATA=20
M_POP=15
NUM_FANTASIES=2
NUM_REF_POINTS=6
# REF_PTS="None"
# "down2-mid-up0" "up0-up1"  "mid-up0-up1" "down2-mid-up0-up1"
# 0.02 0.1
# 0.1 0.2
for T_EDIT in 0.0; do
    for DELTA in 0.2; do
        for RES_BLK in "n"; do #"down2-mid-up0-up1"
            echo "T_EDIT = $T_EDIT"
            echo "DELTA = $DELTA"
            echo "RES_BLK = $RES_BLK"
            echo "Running job for $OBJECTIVES"
            python -u scripts/pbo_manifold_sdxl.py \
            --num_trials $NUM_TRIALS --num_batches $NUM_BATCHES --dim "$DIM" --acf "$ALGOS" --output_path "$RESULT_PATH" \
            --num_initial_samples $NUM_INITIAL_SAMPLES --m $M --num_restarts "$NUM_RESTARTS" --raw_samples $RAW_SAMPLES --q $Q \
            --T "$T" --mode $MODE --num_population_models $NUM_POP_MODELS --m_pop $M_POP --num_population_data $NUM_POP_DATA \
            --logit $LOGIT --multi $MULTI --plotting $PLOTTING --save_results "$SAVE_RESULTS" --obj_name "$OBJECTIVES" \
            --num_fantasies $NUM_FANTASIES --num_ref_points $NUM_REF_POINTS --edit_type $EDIT_TYPE \
            --prompts "$PROMPTS" --t_edit $T_EDIT --delta $DELTA --img_seed $IMG_SEED --res_blk $RES_BLK\
        # --ref_points_path "$REF_PTS"
        done
    done
done