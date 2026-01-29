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
import fire
from utils.demon.src.image_grid import create_image_grid
from utils.demon.pipelines.generate_abstract import DemonGenerater

from PIL import Image


# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

class ChooseGenerator(DemonGenerater):
    def rewards(self, pils):
        """
        Compute a reward for a list of PIL images by creating an image grid.

        Args:
            pils: List of PIL images.

        Returns:
            The generated image grid as the reward.
        """

        new_pils = []
        for pil in pils:
            new_pils.append(pil)
        return create_image_grid(new_pils)
    
@pyrallis.wrap()
def start(opt: DEMON_SDXL_promptset):
    
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
        prompts = prompts[:PROMPTS_PER_CATEGORY]

        if opt.use_random_seeds:
            img_seeds = opt.img_random_seeds[:opt.num_samples_per_prompt]
            print(" ")
            print(f"Using T2I model generation random seeds: {img_seeds}")
            print(" ")
        else:
            img_seeds = [starting_seed+i for i in range(opt.num_samples_per_prompt)]
        # Reset the counter for each category
        counter_id = 0 # in the order acf_algos x trials x img_seeds x num_prompts x categories
        for img_seed in img_seeds:
            for i, prompt in enumerate(prompts):
                print(" ")
                print(f"Generating {i+1}/{len(prompts)} for seed {img_seed} in category {category}")
                print(" ")
                
                opt.img_seed = img_seed
                opt.output_path = os.path.join(root_output_path , category)
                op_path = os.path.join(opt.output_path , "samples") 
                
                opt.prompts = prompt
                print(" ")
                print(f"Prompt: {opt.prompts}")
                print(" ")
                
                image = run_DEMON(opt)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_DEMON(opt: DEMON_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)
    
    
    generator = ChooseGenerator(
        beta=opt.beta,
        tau=opt.tau,
        K=opt.K,
        T=opt.num_inference_steps,
        demon_type=opt.demon_type,
        r_of_c=opt.r_of_c,
        c_steps=opt.c_steps,
        ode_after=opt.ode_after,
        cfg=opt.guidance_scale,
        seed=img_seed,
        save_pils=opt.save_pils,
        experiment_directory=os.path.join(opt.output_path,"config"),
    )

    im = fire.Fire(generator.generate(prompt=prompt))

    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
    