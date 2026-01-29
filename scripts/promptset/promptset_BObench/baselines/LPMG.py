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
import nltk
from nltk.corpus import stopwords
import string
from utils.LPMG.main import LPMConfig, main, setup
from PIL import Image
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
def start(opt: LPMG_promptset):
    
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
                
            image = run_LPMG(opt)
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
                
def run_LPMG(opt: LPMG_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True,).to("cuda")
    comp_words = get_valid_tokens(prompt, pipe.tokenizer)[1]
    
    opt.object_of_interest = random.choice(comp_words)
    
    print(" ")
    print(f"Object of interest -- {opt.object_of_interest}")
    print(" ")
    
    obj_to_preserve = comp_words.copy()
    obj_to_preserve.remove(opt.object_of_interest)
    opt.objects_to_preserve = ",".join(obj_to_preserve)
    
    proxy_words = opt.proxy_words
    objects_to_preserve = opt.objects_to_preserve
    background_nouns = opt.background_nouns
    stable, stable_config = setup(LPMConfig())
    
    prompt = prompt.replace(opt.object_of_interest, '{word}')
    proxy_words = proxy_words.split(',') if proxy_words != '' else []
    objects_to_preserve = objects_to_preserve.split(',') if objects_to_preserve != '' else []
    background_nouns = background_nouns.split(',') if background_nouns != '' else []
    
    args = LPMConfig(
        seed=img_seed,
        prompt=prompt,
        object_of_interest=opt.object_of_interest,
        proxy_words=proxy_words,
        number_of_variations=opt.number_of_variations,
        start_prompt_range=opt.start_prompt_range,
        end_prompt_range=opt.end_prompt_range,
        objects_to_preserve=objects_to_preserve,
        background_nouns=background_nouns,
        real_image_path="" 
    )

    result_images, result_proxy_words = main(stable, stable_config, args)
    result_images = [im.permute(1, 2, 0).cpu().numpy() for im in result_images]
    result_images = [(im * 255).astype(np.uint8) for im in result_images]
    result_images = [Image.fromarray(im) for im in result_images]
    
    result_final = result_images*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
def get_valid_tokens(prompt, tokenizer):
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger_eng")

    tokens = tokenizer(prompt)["input_ids"]
    decoder = tokenizer.decode

    words = [decoder(token) for token in tokens]
    tags = nltk.pos_tag(words)

    selected_indices = []
    selected_words = []
    stop_words = set(stopwords.words('english'))

    for i, (word, pos) in enumerate(tags):
        if i == 0:
            continue
        if word in stop_words or word in string.punctuation or pos.startswith("VB"):
            continue
        selected_indices.append(i)
        selected_words.append(word)
    return selected_indices, selected_words[:-1]