import argparse
import glob
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from utils.clip_musiq_topiq_liqe_lpips_psnr import evaluate_metrics, load_metrics
from PIL import Image
from lavis.models import load_model_and_preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Path to read samples and output scores"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="object_consistency-lpips",
        help="Metric to evaluate [global_iqa-topiq_nr, global_iqa-musiq, global_iqa-liqe, object_consistency-lpips, object_consistency-psnr, background_consistency-lpips, background_consistency-psnr, inpainting_similarity-lpips, inpainting_similarity-psnr, semantic-clip_t2t, semantic-clip_i2i]"
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="./promptsets/COCOEE/cocoee.txt",
        help="Path to ground truth metadata"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="cocoee",
        help="category of dataset"
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="./promptsets/COCOEE",
        help="base dataset directory for src imgs"
    )
    args = parser.parse_args()
    return args


def main(args):

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath=args.outpath
    
    
    image_folder=os.path.join(outpath,'samples')
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort

    cnt = 0
    total = []
    
    with open(args.ref_path, 'r') as f:
        ref_dict = json.load(f)
    ref_cat = ref_dict[args.category]

    for file_name in file_names:
        image_path = os.path.join(image_folder,file_name)
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        prompt = file_name.split("_")[0]
        edited_image = np.array(Image.open(image_path))
        for ref in ref_cat:
            if prompt == ref["prompt"]:
                original_img_path = ref["original_image_path"]
                mask_path = ref["mask_path"]
                max_res = ref["max_resolution"]
                sel_pts = ref["selected_points"]
                words = ref["words"]
                metrics = load_metrics(device)
                blip_model, blip_vis_processors = load_blip_model(device)
                source_image, mask, max_resolution, prompt, selected_points, words = get_data(args.base_data_dir, original_img_path, mask_path, max_res, prompt, words, sel_pts, True)
                
                results_dict = evaluate_metrics(source_image=source_image, edited_image=edited_image,
                                        source_mask=mask, selected_points=selected_points,
                                        metrics=metrics, blip_model=blip_model, blip_vis_processors=blip_vis_processors, device=device)

                reward = get_metric_value(results_dict, args.metric)

        total.append(reward)
        cnt += 1
        if (cnt % 100 == 0):
            print(f"{args.metric}:{cnt} prompt(s) have been processed!")


    sim_dict=[]
    for i in range(len(total)):
        tmp={}
        tmp['question_id']=i
        tmp["answer"] = total[i]
        sim_dict.append(tmp)
    
    json_file = json.dumps(sim_dict)
    savepath = os.path.join(outpath,f"annotation_{args.metric}")
    os.makedirs(savepath, exist_ok=True)
    with open(f'{savepath}/vqa_result.json', 'w') as f:
        f.write(json_file)
    print(f"save to {savepath}")

    # score avg
    score=0
    for i in range(len(sim_dict)):
        score+=float(sim_dict[i]['answer'])
    with open(f'{savepath}/score_avg.txt', 'w') as f:
        f.write('score avg:'+str(score/len(sim_dict)))
    print("score avg:", score/len(sim_dict))


if __name__ == "__main__":
    args = parse_args()
    main(args)



def get_data(data_path, img_path, m_path, max_resolution, prompt, words, selected_points, use_prompt):
    
    
    image_path = os.path.join(data_path, "COCOEE_images", img_path)
    mask_path = os.path.join(data_path, "COCOEE_masks",  m_path)
    
    # load image and mask
    original_image = np.array(exif_transpose(Image.open(image_path)).convert('RGB'))
    mask = np.array(exif_transpose(Image.open(mask_path)).convert('L'))
    # Resize image and mask
    h, w = Image.fromarray(original_image).size
    factor = max_resolution / (min(h, w))
    if factor != 1:
        h, w = int(h * factor), int(w * factor)
        original_image = np.array(Image.fromarray(original_image).resize((h, w), Image.BICUBIC))
        mask = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST))
    mask = np.expand_dims(mask, axis=2)
        
    if use_prompt:
        return original_image, mask, max_resolution, prompt, selected_points, words
    else:
        return original_image, mask, max_resolution, "", selected_points, []


def exif_transpose(img):
    if not img:
        return img 
    exif_orientation_tag = 274

    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        if orientation == 1:
            pass 
        elif orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180)
        elif orientation == 4:
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    return img

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def load_blip_model(device):
    blip_model, blip_vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    return blip_model, blip_vis_processors

def get_metric_value(results, key_string):
    # Split the string into ['global_iqa', 'topiq_nr']
    keys = key_string.split("-")
    
    # Traverse the dictionary
    value = results
    for k in keys:
        value = value[k]
    return value