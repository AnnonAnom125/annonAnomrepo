# !/bin/bash
# catch arguments

#-------------------------------------------------------------------
#             Evaluate on AE prompts
#-------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=2
sample_root="./outputs/promptset/promptset_ae/DiffDPO/low-image/sdxl"

categories="animals animals_objects objects"

for category in $categories; do
    echo "Evaluating VILA on ${sample_root}/${category}/"
    python scripts/evaluation/evaluate_vila.py --outpath "${sample_root}/${category}/"
done