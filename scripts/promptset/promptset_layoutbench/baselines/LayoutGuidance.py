import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..","..")))
import json
import pyrallis
import numpy as np
import torch
import random
from diffusers import AutoencoderKL, LMSDiscreteScheduler, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from configs import *

from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from utils.layoutguidance import unet_2d_condition
from utils.layoutguidance.inp import load_text_inversion, compute_ca_loss, Pharse2idx, draw_box, setup_logger
from tqdm import tqdm
from omegaconf import OmegaConf

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: LayoutGuidance_promptset):
    
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
                
                opt.prompts = prompt_ed_lst["caption"]
                prompt = prompt_ed_lst["caption"]
                
                bboxes = prompt_ed_lst["boxes_normalized"]
                box_cap = prompt_ed_lst["box_captions"]
                
                print(" ")
                print(f"Prompt: {opt.prompts}")
                print(" ")
                
                image = run_LayoutGuidance(opt, bboxes)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_LayoutGuidance(opt: LayoutGuidance_promptset, bboxes, box_cap):
    
    prompt = opt.prompts 
    bboxes = [bboxes]
    phrases = ";".join(box_cap)
    
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    cfg = OmegaConf.load(opt.base_config) 
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg)
    
    im = pil_images[0]
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    

def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg):



    # Get Object Positions

    object_positions = Pharse2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the initial latent noise

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)

            # update latents with guidance
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions) * cfg.inference.loss_scale

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images    