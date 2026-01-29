#!/bin/sh

# Accept arguments
CUDA_VISIBLE_DEVICES="$1"
OBJECTIVES="$2"
MODE="$3"
ALGOS="$4" 
RESULT_PATH="$5"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

NUM_TRIALS=2
NUM_BATCHES=15
DIM=4
NUM_INITIAL_SAMPLES=10
M=10
NUM_RESTARTS=5 #3
RAW_SAMPLES=512
Q=4
T=4
NUM_POP_MODELS=2
NUM_POP_DATA=10
M_POP=15
LOGIT="True"
MULTI="True"
PLOTTING="True"
SAVE_RESULTS="True"
NUM_FANTASIES=2
NUM_REF_POINTS=6
# REF_PTS="None"




echo "Running job for $OBJECTIVES"
python -u scripts/multi_pbo_synthetic.py \
--num_trials $NUM_TRIALS --num_batches $NUM_BATCHES --dim "$DIM" --acf "$ALGOS" --output_path "$RESULT_PATH" \
--num_initial_samples $NUM_INITIAL_SAMPLES --m $M --num_restarts "$NUM_RESTARTS" --raw_samples $RAW_SAMPLES --q $Q \
--T "$T" --mode $MODE --num_population_models $NUM_POP_MODELS --m_pop $M_POP --num_population_data $NUM_POP_DATA \
--logit $LOGIT --multi $MULTI --plotting $PLOTTING --save_results "$SAVE_RESULTS" --obj_name "$OBJECTIVES" \
--num_fantasies $NUM_FANTASIES --num_ref_points $NUM_REF_POINTS \

# --ref_points_path "$REF_PTS"