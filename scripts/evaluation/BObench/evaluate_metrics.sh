# !/bin/bash
# catch arguments
device="$1"
sample_root="$2"
categories="$3"

METRICS="lpips clipi vila picscore hpsv2 aesthetic imagereward"  
#picscore clip hpsv2 aesthetic imagereward topiq musiq liqe psnr lpips ssim clipi dino vila

export CUDA_VISIBLE_DEVICES=$device
root_dir="${sample_root}"


for category in $categories; do
    for metric in $METRICS; do
        echo "Evaluating $metric on $root_dir/${category}/"
        python scripts/evaluation/BObench/evaluate_metrics.py --outpath "$root_dir/${category}/" --metric "$metric"
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