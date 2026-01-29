import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..","..")))
import json
import pyrallis
import numpy as np
import torch
import random
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from configs import *
import re
from utils.obj_utils import get_objective
import importlib

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def sanitize_filename(text):
    return re.sub(r'[^\w\-_\. ]', '', text).replace(' ', '_')[:100]

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{sanitize_filename(opt.prompts)}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: DiffDPO_SDXL_promptset):
    
    prompt_path = opt.promptset_path 
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)

    prompt_categories = list(prompt_dict.keys())
    PROMPTS_PER_CATEGORY = opt.prompts_per_category
    root_output_path = opt.output_path 
    starting_seed = opt.seed

    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path,exist_ok=True)

    for category in prompt_categories:
        prompts = prompt_dict[category]
        if opt.prompts_per_category == -1:
            PROMPTS_PER_CATEGORY = len(prompts)
        prompts_ids = prompts[:PROMPTS_PER_CATEGORY]

        # Reset the counter for each category
        counter_id = 0 # in the order acf_algos x trials x img_seeds x num_prompts x categories
        for i, prompt_idx in enumerate(prompts_ids):
            print(" ")
            print(f"Generating {i+1}/{len(prompts)} in category {category}")
            print(" ")
            
            img_seed = int(prompt_idx["edit_seed"])
            tar_seed = int(prompt_idx["target_seed"])
            prompt = prompt_idx["prompt"]
            prompt_total = []
            
            _, _, obj_name = get_objective(opt.obj_name,opt.mode)
            obj = importlib.import_module(obj_name)
            print(" ")
            print(f"Generating Target")
            print(" ")
            target_img = obj.unedited_generation(prompt,tar_seed,opt.num_inference_steps,"cuda")
            
            opt.img_seed = img_seed
            opt.output_path = os.path.join(root_output_path , category)
            op_path = os.path.join(opt.output_path , "samples")
            ref_path = os.path.join(opt.output_path , "reference")
            tar_path = os.path.join(opt.output_path , "target")
            prompt_tot_path = os.path.join(opt.output_path , "prompts.json")
            os.makedirs(op_path, exist_ok=True)
            os.makedirs(ref_path, exist_ok=True)
            os.makedirs(tar_path, exist_ok=True)
            
            opt.prompts = prompt
            print(" ")
            print(f"Prompt: {opt.prompts}")
            print(" ")
            image = run_diffDPO(opt)
            # image is in the order of acf_algos x trials
            for i in range(len(image)):
                save_path = os.path.join(op_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                save_tar_path = os.path.join(tar_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                image[i].save(save_path)
                target_img.save(save_tar_path)
                prompt_total.append(opt.prompts)
                print(" ")
                counter_id += 1
        with open(prompt_tot_path, "w") as f:
            json.dump(prompt_total, f)
                
def run_diffDPO(opt: DiffDPO_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)
    dpo_unet = UNet2DConditionModel.from_pretrained(
                        'mhdang/dpo-sdxl-text2image-v1',
                        subfolder='unet',
                        torch_dtype=torch.float16
                        ).to('cuda')
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pipe.safety_checker = None

    generator = torch.Generator(device='cuda')
    pipe.unet = dpo_unet
    generator = generator.manual_seed(img_seed)
        
    im = pipe(prompt=prompt, generator=generator, num_inference_steps=opt.num_inference_steps).images[0]
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
    