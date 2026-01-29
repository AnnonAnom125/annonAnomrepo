import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import json
import pyrallis
import numpy as np
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch
import random
import importlib

from matplotlib import pyplot as plt

import torch.nn.functional as F
from utils.obj_utils import get_objective
from configs import *
import re
from PIL import Image

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")


def create_folder_path(opt, tar_seed):
    if opt.multi:
        pp = os.path.join(opt.output_path,f"Multiwise")
    else:
        pp = os.path.join(opt.output_path,f"Pairwise")
        if opt.T != 2:
            opt.convert_to_pair = True
        else:
            opt.convert_to_pair = False
            assert opt.T == 2
    
    if opt.obj_name not in opt.image_models:
        pp = os.path.join(pp,f"dim-{opt.dim}")

    if opt.logit:
        opt.type_likelihood=True
        pp = os.path.join(pp,f"Logit-{opt.acf}")
    else:
        pp = os.path.join(pp,f"Probit-{opt.acf}")
        if not opt.multi:
            opt.type_likelihood=False
        else:
            raise ValueError(f"No probit model for MultiwiseGP")
    
    if opt.obj_name in opt.image_models:
        pp = os.path.join(pp,f"{sanitize_filename(opt.prompts)}",f"seed-{opt.img_seed}",f"tar-seed-{tar_seed}")
    return pp

def sanitize_filename(text):
    return re.sub(r'[^\w\-_\. ]', '', text).replace(' ', '_')[:100]

@pyrallis.wrap()
def start(opt: MultiBOConfig_SDXL_promptset):
    
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
            if category == "part2" and i < 3:
                counter_id += opt.num_trials
            else:
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
            
                image, ref_paths = run_l2(opt, target_img, tar_seed)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                    save_ref_path = os.path.join(ref_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                    save_tar_path = os.path.join(tar_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    r_img = Image.open(ref_paths[i])
                    r_img.save(save_ref_path)
                    target_img.save(save_tar_path)
                    prompt_total.append(opt.prompts)
                    print(" ")
                    counter_id += 1
        with open(prompt_tot_path, "w") as f:
            json.dump(prompt_total, f)
                
def run_l2(opt: MultiBOConfig_SDXL_promptset, target_img=None, tar_seed=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.output_path = create_folder_path(opt, tar_seed)
    # 1. Pipeline Setup (Standard SDXL)
    # Ensure torch_dtype matches the target_latents for precision consistency
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
    ).to(device)
    
    # 2. Encode Target for L2 Loss
    target_img_res = target_img.resize((1024, 1024))
    img_t = torch.from_numpy(np.array(target_img_res)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)
    img_t = (img_t / 127.5) - 1.0
    with torch.no_grad():
        target_latents = pipe.vae.encode(img_t).latent_dist.sample() * pipe.vae.config.scaling_factor

    # 3. Guidance Callback
    guidance_mag = getattr(opt, 'guidance_mag', 1.0)

    def dual_path_callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        # Ensure both paths start from identical noise at the first step
        if step_index == 0:
            latents[1:2] = latents[0:1].clone()

        with torch.enable_grad():
            # 1. Clone and enable grad on the 2nd path
            guided_latent = latents[1:2].detach().requires_grad_(True)
            
            # 2. Compute L2 Loss
            loss = F.mse_loss(guided_latent, target_latents)
            grad = torch.autograd.grad(loss, guided_latent)[0]
            
            # 3. Normalize Gradient (Crucial to prevent black images/NaNs)
            # This keeps the "nudge" stable across all steps
            grad = grad / (grad.norm() + 1e-8)
            
            # 4. Apply a smaller, stable guidance magnitude
            # Start with 0.1 and adjust as needed
            stable_mag = 0.5 
            latents[1:2] = latents[1:2].detach() - (stable_mag * grad)
        
        callback_kwargs["latents"] = latents
        return callback_kwargs


    # 4. Inference
    generator = torch.Generator(device=device).manual_seed(opt.img_seed)
    
    # num_images_per_prompt=2 creates the reference and guided paths in one batch
    output = pipe(
        prompt=opt.prompts,
        num_inference_steps=opt.num_inference_steps,
        num_images_per_prompt=2,
        generator=generator,
        callback_on_step_end=dual_path_callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )

    # 5. Handle Reference Image Saving
    ref_dir = os.path.join(opt.output_path, "image_orig")
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "image_orig.png")
    
    # Save the unedited path (index 0) if it doesn't exist
    if not os.path.exists(ref_path):
        output.images[0].save(ref_path)



    return [output.images[1]]*opt.num_trials, [ref_path]*opt.num_trials

if __name__ == "__main__":
    start()
    
    