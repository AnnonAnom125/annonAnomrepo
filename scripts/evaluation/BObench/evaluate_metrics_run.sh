# !/bin/bash
# catch arguments
#  lpips clipi vila picscore hpsv2 aesthetic imagereward
#-------------------------------------------------------------------
#             Evaluate on BObench prompts
#-------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=0
sample_root="./outputs/promptset/promptset_BObench/MultiBO/low-image/sdxl/final"

categories="part7"

metrics="lpips clipi vila picscore hpsv2 aesthetic imagereward"
for category in $categories; do
    for metric in $metrics; do
        echo "Evaluating ${metric} on ${sample_root}/${category}/"
        python scripts/evaluation/BObench/evaluate_metrics.py --outpath "${sample_root}/${category}/" --metric "${metric}"
    done
done