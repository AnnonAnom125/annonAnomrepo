
gpu1=0
gpu2=1
gpu3=5

# ## ----------------------------------------
##              Evaluation on AE set 
## --------------------------------------------

np_num=4
categories="rarebench_multi_3complex"
#  rarebench_multi_3complex"
# categories="complex_val"
# categories="objects"

PID=2930664      # job currently running

echo "Waiting for PID $PID to finish..."
while kill -0 $PID 2>/dev/null; do
    sleep 10
done

echo "Job finished. Starting next script."

sample_root="outputs/output_eval_full/rarebench/PixART/cfg5.0_"
# Any prompset
./scripts/evaluation/evaluate_picscore_aes_clip_hpsv2_imagereward.sh $gpu2 $sample_root "$categories"
./scripts/evaluation/evaluate_vila.sh $gpu2 $sample_root "$categories"
./scripts/evaluation/BLIPvqa_eval/evaluate_blip_vqa_base.sh $gpu2 $sample_root "$categories" "${np_num}"

# COCOEE
categories="cocoee"
./scripts/evaluation/evaluate_clip_musiq_topiq_liqe_lpips_psnr.sh $gpu2 $sample_root "$categories"
# AnyBench
categories="anybench"
./scripts/evaluation/evaluate_clipi_clipt_dino_lpips.sh $gpu2 $sample_root "$categories"

