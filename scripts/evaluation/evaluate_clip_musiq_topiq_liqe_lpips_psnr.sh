# !/bin/bash
# catch arguments
device="$1"
sample_root="$2"
categories="$3"

METRICS="global_iqa-topiq_nr global_iqa-musiq global_iqa-liqe object_consistency-lpips object_consistency-psnr background_consistency-lpips background_consistency-psnr inpainting_similarity-lpips inpainting_similarity-psnr semantic-clip_t2t semantic-clip_i2i"  

export CUDA_VISIBLE_DEVICES=$device
root_dir="${sample_root}"


for category in $categories; do
    for metric in $METRICS; do
        echo "Evaluating $metric on $root_dir/${category}/"
        python scripts/evaluation/evaluate_clip_musiq_topiq_liqe_lpips_psnr.py --outpath "$root_dir/${category}/" --metric "$metric" --category "$caregory"
    done
done 




#----- t2icomp -----------

# categories=(color_val shape_val texture_val)


# #----------------- for t2istyle -----------------
# for category in ${categories[@]}; do
#     sample_root="outputs/output_eval_full/t2icompbench/SDXL/cfg5.0_/${category}/"
#     echo $sample_root
#     python scripts/evaluate_imageReward_t2icomp.py --outpath $sample_root 
# done