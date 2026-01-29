# !/bin/bash
# catch arguments

#-------------------------------------------------------------------
#             Evaluate on COCOEE
#-------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=2
sample_root="./outputs/promptset/promptset_ae/DiffDPO/low-image/sdxl"

categories="cocoee"

metrics="global_iqa-topiq_nr global_iqa-musiq global_iqa-liqe object_consistency-lpips object_consistency-psnr background_consistency-lpips background_consistency-psnr inpainting_similarity-lpips inpainting_similarity-psnr semantic-clip_t2t semantic-clip_i2i"  
for category in $categories; do
    for metric in $metrics; do
        echo "Evaluating ${metric} on ${sample_root}/${category}/"
        python scripts/evaluation/evaluate_picscore_aes_clip_hpsv2_imagereward.py --outpath "${sample_root}/${category}/" --metric "${metric}"
    done
done