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
from PIL import Image
import clip
from utils.clipi_clipt_dino_lpips.anybench_utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black
from torchvision import transforms

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
        help="Metric to evaluate [clip_i, clip_t, dino, l1]"
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="./promptsets/AnyBench-test/",
        help="Path to ground truth metadata"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="movement",
        help="category of dataset"
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="./promptsets/AnyBench-test/",
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
    
    ref_path = os.path.join(args.ref_path,f"{args.category}.txt")
    with open(ref_path, 'r') as f:
        ref_dict = json.load(f)
    ref_cat = ref_dict[args.category]

    for file_name in file_names:
        image_path = os.path.join(image_folder,file_name)
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        prompt = file_name.split("_")[0]
        for ref in ref_cat:
            if prompt == ref["input"]:
                original_img_path = ref["image_file"]
                mask_path = ref["mask_path"]
                output = ref["output"]

                if not is_all_black(image_path):
                    image_pairs=[Image.open(original_img_path),  # gt
                      Image.open(image_path).convert('RGB'),  # output
                     output]
            
                clip_model, transform = clip.load("ViT-B/32")
                clip_i = eval_clip_i(image_pairs=image_pairs, model=clip_model, transform=transform, url_flag=False)
                if output is not None:
                    clip_t, _ = eval_clip_t(image_pairs=image_pairs, model=clip_model, transform=transform, url_flag=False)
                else:
                    clip_t = -100
                l1 = eval_distance(image_pairs=image_pairs, metric='l1', url_flag=False)

                dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                dino_model.eval()
                dino_model.to(device)
                dino_transform = transforms.Compose([
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                dino_score = eval_clip_i(image_pairs=image_pairs, model=dino_model, transform=dino_transform,
                                         url_flag=False, metric='dino')
                result_dict = {
                    "clip_i": clip_i, "clip_t": clip_t,  "dino": dino_score, "l1": l1
                }
                reward = result_dict[args.metric]
                
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


