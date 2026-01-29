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
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from utils.pic_aes_clip_hpsv2 import clip_utils, aes_utils, hps_utils, pickscore_utils
import ImageReward as RM

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
        default="picscore",
        help="Metric to evaluate [clip, hpsv2, aes, picscore]"
    )
    parser.add_argument(
        "--aes_model_path",
        type=str,
        default="./models/checkpoints/sac+logos+ava1-l14-linearMSE.pth",
        help="Path to model aes"
    )
    parser.add_argument(
        "--hps_model_path",
        type=str,
        default="./models/checkpoints/HPS_v2_compressed.pt",
        help="Path to model hpsv2"
    )
    parser.add_argument(
        "--prompt_in_filename",
        type=bool,
        default=False,
        help="is the full prompt in the filename or truncated"
    )
    args = parser.parse_args()
    return args


def main(args):

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath=args.outpath
    if args.metric == "clip":
        model = clip_utils.Selector(device)
    elif args.metric == "aesthetic":
        model = aes_utils.Selector(device, args.aes_model_path)
    elif args.metric == "hpsv2":
        model = hps_utils.Selector(device, args.hps_model_path)
    elif args.metric == "picscore":
        model = pickscore_utils.Selector(device)
    elif args.metric == "imagereward":
        model = RM.load("ImageReward-v1.0")
        model = model.to(device)
        

    image_folder=os.path.join(outpath,'samples')
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort
    counter_ids = [int(x.split("_")[-1].split('.')[0]) for x in file_names]

    cnt = 0
    total = []

    for file_name, i in zip(file_names, counter_ids):
        image_path = os.path.join(image_folder,file_name)
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        if args.prompt_in_filename:
            prompt = file_name.split("_")[0]
        else:
            prompt_path = os.apth.join(outpath, "prompts.json")
            with open(prompt_path, 'r') as f:
                prompt_list = json.load(f)
            prompt = prompt_list[i]
            
        with torch.no_grad():
            if args.metric == "imagereward":
                reward = model.score(prompt, [image_path])
            else:
                reward = model.score([image_path], prompt)[0]


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


    # model = RM.load("ImageReward-v1.0")
    # model = model.to(device)

    # scores = {}
    # for path, cap, imgid in tqdm(zip(image_paths, captions, image_ids)):
    #     reward = model.score(cap, [path])
    #     scores[imgid] = reward
    
    
    # scores, mean_score, std_score = imageRewardEval(args.sample_root, args.device)
    # print(f"Calculated scores for {len(scores)} images...")
    # print('ImageRewardScore: {:.4f}'.format(mean_score))
    
    # with open(outpath, 'w') as f:
    #     f.write(f"Results for {args.sample_root}:\n")
    #     f.write("Mean score: {:.4f}\t Std: {:.5f}\n".format(mean_score, std_score))
    
    # # Save scores to csv file
    # score_file = os.path.join(args.sample_root, 'imageReward_scores.csv')
    # df = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    # df.index.name = 'image_id'
    # df.to_csv(score_file)
    # print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)