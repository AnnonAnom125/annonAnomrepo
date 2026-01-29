# !/bin/bash
# catch arguments
device="$1"
sample_root="$2"
categories="$3"

export CUDA_VISIBLE_DEVICES=$device
root_dir="${sample_root}"


for category in $categories; do
    echo "Evaluating VILA on $root_dir/${category}/"
    python scripts/evaluation/evaluate_vila.py --outpath "$root_dir/${category}/"
done 




#----- t2icomp -----------

# categories=(color_val shape_val texture_val)


# #----------------- for t2istyle -----------------
# for category in ${categories[@]}; do
#     sample_root="outputs/output_eval_full/t2icompbench/SDXL/cfg5.0_/${category}/"
#     echo $sample_root
#     python scripts/evaluate_imageReward_t2icomp.py --outpath $sample_root 
# done