# !/bin/bash
# catch arguments

#-------------------------------------------------------------------
#             Evaluate on AnyBench
#-------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=2
sample_root="./outputs/promptset/promptset_ae/DiffDPO/low-image/sdxl"

categories="movement resize"

metrics="clip_i clip_t dino l1"
for category in $categories; do
    for metric in $metrics; do
        echo "Evaluating ${metric} on ${sample_root}/${category}/"
        python scripts/evaluation/BObench/evaluate_clipi_clipt_dino_lpips.py --outpath "${sample_root}/${category}/" --metric "${metric}"
    done
done