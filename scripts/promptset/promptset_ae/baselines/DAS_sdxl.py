import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..","..")))
import json
import pyrallis
import numpy as np
import torch
import random
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from configs import *
from utils.das import rewards
from utils.das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: DAS_SDXL_promptset):
    
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
                
                image = run_DAS(opt)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_DAS(opt: DAS_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    repeated_prompts = [prompt] * opt.batch_p

    if opt.reward_type == "aesthetic":
        reward_fn = rewards.aesthetic_score(device = 'cuda')
    else:
        reward_fn = rewards.PickScore(device = 'cuda')
    image_reward_fn = lambda images: reward_fn(
                        images, 
                        repeated_prompts
                    )

    ################### Initialize ###################

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(opt.num_inference_steps)
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)

    ################### Inference ###################
    im = pipeline_using_smc_sdxl(
        pipe,
        generator=generator,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=opt.num_inference_steps,
        output_type="pil",
        # SMC parameters
        num_particles=opt.num_particles,
        batch_p=opt.batch_p,
        tempering_gamma=opt.tempering_gamma,
        reward_fn=image_reward_fn,
        kl_coeff=opt.kl_coeff,
    ).images[0]
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
    