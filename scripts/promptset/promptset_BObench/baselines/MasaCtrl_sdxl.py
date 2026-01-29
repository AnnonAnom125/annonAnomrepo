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
from utils.masactrl.masactrl_utils import AttentionBase
from utils.masactrl.masactrl_utils import regiter_attention_editor_diffusers
from utils.masactrl.masactrl import MutualSelfAttentionControl
import re

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def sanitize_filename(text):
    return re.sub(r'[^\w\-_\. ]', '', text).replace(' ', '_')[:100]
def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: MasaCtrl_SDXL_promptset):
    
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
                
            image = run_MasaCtrl(opt)
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
                
def run_MasaCtrl(opt: MasaCtrl_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    # model_path = "Linaqruf/animagine-xl"
    scheduler = DDIMScheduler(beta_start=opt.beta_start, beta_end=opt.beta_end, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = StableDiffusionXLPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

    set_all_seeds(img_seed)
    
    comp_words = ['turned left', 'turned right']
    
    interest = random.choice(comp_words)
    
    print(" ")
    print(f"Truned to -- {interest}")
    print(" ")
    
    prompts = [prompt]*2
    prompts[1] = prompts[1] + interest

    STEP = opt.step
    LAYER_LIST = opt.layer_list  # run the synthesis with MasaCtrl at three different layer configs

    # initialize the noise map
    start_code = torch.randn([1, 4, 128, 128], device=device)
    # start_code = None
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # inference the synthesized image without MasaCtrl
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    # image_ori = model(prompts, latents=start_code, guidance_scale=7.5).images

    for LAYER in LAYER_LIST:
        # hijack the attention module
        editor = MutualSelfAttentionControl(STEP, LAYER, model_type="SDXL")
        regiter_attention_editor_diffusers(model, editor)

        # inference the synthesized image
        image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5).images

    im = image_masactrl[-1]
        
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
def set_all_seeds(seed):
    """
    Sets the random seed for numpy, random, and pytorch (CPU and CUDA).
    Also sets environment variable for Python hash seed and CUDNN for deterministic behavior.
    """
    # os.environ['PYTHONHASHSEED'] = str(seed) # Set Python hash seed
    random.seed(seed) # Set Python's built-in random module seed
    np.random.seed(seed) # Set NumPy's random seed
    torch.manual_seed(seed) # Set PyTorch's CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set PyTorch's current GPU random seed
        torch.cuda.manual_seed_all(seed) # Set PyTorch's all GPUs random seed
    
    # For deterministic behavior with CUDA operations (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  