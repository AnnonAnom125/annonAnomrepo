# !/bin/bash
# catch arguments

#-------------------------------------------------------------------
#             Evaluate on AE prompts
#-------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=2
sample_root="./outputs/promptset/promptset_ae/DiffDPO/low-image/sdxl"

categories="animals animals_objects objects"

metrics="picscore clip hpsv2 aesthetic imagereward"
for category in $categories; do
    for metric in $metrics; do
        echo "Evaluating ${metric} on ${sample_root}/${category}/"
        python scripts/evaluation/evaluate_picscore_aes_clip_hpsv2_imagereward.py --outpath "${sample_root}/${category}/" --metric "${metric}"
    done
done