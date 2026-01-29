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
from utils.pixelman.src.demo.model import EditModels
from PIL import Image
import ast

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: PixelMan_promptset):
    
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
        prompts_edit_dict = prompt_dict[category]
        if opt.prompts_per_category == -1:
            PROMPTS_PER_CATEGORY = len(prompts_edit_dict)
        prompts_edit_dict = prompts_edit_dict[:PROMPTS_PER_CATEGORY]

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
            for i, prompt_ed_lst in enumerate(prompts_edit_dict):
                print(" ")
                print(f"Generating {i+1}/{len(prompts_edit_dict)} for seed {img_seed} in category {category}")
                print(" ")
                
                opt.img_seed = img_seed
                opt.output_path = os.path.join(root_output_path , category)
                op_path = os.path.join(opt.output_path , "samples")
                
                opt.prompts = prompt_ed_lst["prompt"]
                prompt = prompt_ed_lst["prompt"]
    
                opt.max_resolution = prompt_ed_lst["max_resolution"]
                opt.words = prompt_ed_lst["words"]
                opt.selected_points = prompt_ed_lst["selected_points"]
                
                image_path = os.path.join(opt.COCOEE_path, "COCOEE_images", prompt_ed_lst["original_image_path"])
                mask_path = os.path.join(opt.COCOEE_path, "COCOEE_masks",  prompt_ed_lst["mask_path"])
                
                # load image and mask
                original_image = np.array(exif_transpose(Image.open(image_path)).convert('RGB'))
                mask = np.array(exif_transpose(Image.open(mask_path)).convert('L'))
                # Resize image and mask
                h, w = Image.fromarray(original_image).size
                factor = opt.max_resolution / (min(h, w))
                if factor != 1:
                    h, w = int(h * factor), int(w * factor)
                    original_image = np.array(Image.fromarray(original_image).resize((h, w), Image.BICUBIC))
                    mask = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST))
                mask = np.expand_dims(mask, axis=2)
                    
                
                print(" ")
                print(f"Prompt: {opt.prompts}")
                print(" ")
                
                image = run_pixelman(opt, original_image, mask)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_pixelman(opt: PixelMan_promptset, original_image, mask):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    
    model = EditModels(pretrained_model_path="runwayml/stable-diffusion-v1-5", steps=opt.steps, use_ip_adapter=False)
    specific = ast.literal_eval(opt.coefficients)
    w_edit, w_content, w_contrast, w_inpaint = specific
    im = model.run_move(original_image, 
                            mask, 
                            mask_ref=None, 
                            prompt=prompt, 
                            resize_scale=1, 
                            w_edit=w_edit, 
                            w_content=w_content, 
                            w_contrast=w_contrast, 
                            w_inpaint=w_inpaint, 
                            seed=img_seed, 
                            selected_points=opt.selected_points, 
                            guidance_scale=4, 
                            energy_scale=0.5, 
                            max_resolution=opt.max_resolution, 
                            SDE_strength=0.4, 
                            ip_scale=0.1,
                            use_gsn=bool(opt.use_gsn), 
                            inversion_free=bool(opt.inversion_free), 
                            sa_masking_ipt=bool(opt.sa_masking_ipt), 
                            use_copy_paste=bool(opt.use_copy_paste),
                            )[0]
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
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