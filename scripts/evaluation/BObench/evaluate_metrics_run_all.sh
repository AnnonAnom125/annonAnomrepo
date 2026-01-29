#!/bin/bash
# 1. Base path configuration
base_path="/home/exx/Documents/raja/MultiBO/outputs/promptset/promptset_BObench"

# 2. List of methods (including the plain MultiBO)

    # "MultiBO"
    
    # "MultiBO-non-human_aesthetic"
    # "MultiBO-non-human_picscore"
    # "MultiBO-non-human_clip"
    # "MultiBO-non-human_hpsv2"
    # "MultiBO-non-human_imagereward"
methods=(
    "MultiBO"
)

export CUDA_VISIBLE_DEVICES=1
categories="part7"
metrics="lpips clipi vila picscore hpsv2 aesthetic imagereward"

for method in "${methods[@]}"; do
    # 3. Handle the "MultiBO" path exception
    if [ "$method" == "MultiBO" ]; then
        # Path: base/MultiBO/low-image/sdxl/final/category
        sample_root="${base_path}/${method}/low-image/sdxl/final"
    else
        # Path: base/MethodName/low-image/sdxl/category
        sample_root="${base_path}/${method}/low-image/sdxl"
    fi
    
    for category in $categories; do
        target_path="${sample_root}/${category}/"
        
        # Check if the directory exists before running to avoid python errors
        if [ -d "$target_path" ]; then
            for metric in $metrics; do
                echo "Processing: $method | $category | $metric"
                
                python scripts/evaluation/BObench/evaluate_metrics.py \
                    --outpath "$target_path" \
                    --metric "$metric"
            done
        else
            echo "Skipping: $target_path (Directory not found)"
        fi
    done
done
