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
from scripts.promptset.promptset_BObench.baselines.utils.demon.src.image_grid import create_image_grid
from scripts.promptset.promptset_BObench.baselines.utils.demon.pipelines.generate_abstract import DemonGenerater
import re
from PIL import Image
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

# Lazy model getters.
_aesthetic_scorer = None
def get_aesthetic_scorer():
    global _aesthetic_scorer
    if _aesthetic_scorer is None:
        from scripts.promptset.promptset_BObench.baselines.utils.demon.src.reward_models.AestheticScorer import AestheticScorer
        _aesthetic_scorer = AestheticScorer().to('cuda')
    return _aesthetic_scorer

_pickscore_processor = None
_pickscore_model = None
def get_pickscore_models():
    global _pickscore_processor, _pickscore_model
    if _pickscore_processor is None or _pickscore_model is None:
        _pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        _pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')
    return _pickscore_processor, _pickscore_model

_imageReward_model = None
def get_imageReward_model():
    global _imageReward_model
    if _imageReward_model is None:
        # Import ImageReward lazily.
        RM = importlib.import_module("ImageReward")
        _imageReward_model = RM.load("ImageReward-v1.0")
    return _imageReward_model

@torch.inference_mode()
def hpsv2_reward(pil):
    """Compute reward using hpsv2 and scale it appropriately."""
    return hpsv2.score(pil, prompt, hps_version="v2.1")[0] * 40

@torch.inference_mode()
def rm_reward(pil):
    """Compute reward using ImageReward."""
    model = get_imageReward_model()
    return model.score(prompt, [pil])

@torch.inference_mode()
def pickscore_reward(pil):
    """Compute reward using PickScore."""
    processor, model = get_pickscore_models()
    inputs = processor(images=pil, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return model(**inputs).logits_per_image.item()

@torch.inference_mode()
def aesthetic_reward(pil):
    """Compute aesthetic reward."""
    scorer = get_aesthetic_scorer()
    return scorer(pil).item()

def reward(pil):
        total = 0
        aesthetic = True
        imagereward = False
        pickscore = False
        hpsv2_flag = False
        if aesthetic:
            total += aesthetic_reward(pil)
        if imagereward:
            total += rm_reward(pil)
        if pickscore:
            total += pickscore_reward(pil)
        if hpsv2_flag:
            total += hpsv2_reward(pil)
        return total

class QualitativeGenerater(DemonGenerater):
    def rewards(self, pils):
        """Compute rewards for each generated PIL image."""
        return [reward(pil) for pil in pils]
    
    def generate(self, prompt):
        """
        Override generate to update the config file with reward flags.
        Uses ODE-only sampling if no reward model is active.
        """
        aesthetic=True,
        imagereward=False,
        pickscore=False,
        hpsv2_flag=False,
        super().generate(prompt, ode=not any([aesthetic, imagereward, pickscore, hpsv2_flag]))
        # Update config file with reward options.
        config_path = f'{self.log_dir}/config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        config.update({
            'aesthetic': aesthetic,
            'imagereward': imagereward,
            'pickscore': pickscore,
            'hpsv2': hpsv2_flag
        })
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

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
                
            image, refs= run_DEMON(opt)
            # image is in the order of acf_algos x trials
            for i in range(len(image)):
                save_path = os.path.join(op_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png")  
                save_ref_path = os.path.join(ref_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                save_tar_path = os.path.join(tar_path, f"{sanitize_filename(prompt)}_{counter_id:06d}.png") 
                image[i].save(save_path)
                refs[i].save(save_ref_path)
                target_img.save(save_tar_path)
                prompt_total.append(opt.prompts)
                print(" ")
                counter_id += 1
        with open(prompt_tot_path, "w") as f:
            json.dump(prompt_total, f)
                
def run_DEMON(opt: DEMON_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)
    
    
    # generator = ChooseGenerator(
    #     beta=opt.beta,
    #     tau=opt.tau,
    #     K=opt.K,
    #     T=opt.num_inference_steps,
    #     demon_type=opt.demon_type,
    #     r_of_c="baseline",
    #     c_steps=opt.c_steps,
    #     ode_after=opt.ode_after,
    #     cfg=2,
    #     seed=img_seed,
    #     save_pils=opt.save_pils,
    #     experiment_directory=os.path.join(opt.output_path,"config"),
    # )
    generator = QualitativeGenerater(
        beta=0.1,
        tau="adaptive",
        K=4,
        T=50,
        demon_type="tanh",
        r_of_c="baseline",
        c_steps=20,
        ode_after=0.11,
        cfg=2,
        seed=img_seed,
        save_pils=False,
        experiment_directory=os.path.join(opt.output_path),
    )
    generator.generate(prompt=prompt)
    # im = fire.Fire(generator.generate(prompt=prompt))

    result_final = [Image.open(os.path.join(opt.output_path, 'out.png'))]*opt.num_trials
    return result_final, [Image.open(os.path.join(opt.output_path, 'init.png'))]*opt.num_trials

if __name__ == "__main__":
    start()
    
    