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

import torch.nn as nn
import torchvision
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import argparse
import torch.utils.checkpoint as checkpoint
import shutil
from PIL import Image
import time
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from utils.dno.rewards import RFUNCTIONS


# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

# sampling algorithm
class SequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
   
        model_kwargs = {
            "sample": torch.stack([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t, t], device = self.device),
            "encoder_hidden_states": prompt_embeds
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)
        direction = direction[0]

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]

def sequential_sampling(pipeline, unet, sampler, prompt_embeds, added_cond_kwargs, noise_vectors): 


    sampler.initialize(noise_vectors)

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        #model_output = pipeline.unet(**model_kwargs)
        model_output = checkpoint.checkpoint(unet, model_kwargs["sample"], model_kwargs["timestep"], model_kwargs["encoder_hidden_states"], None, None, None, None, added_cond_kwargs)
        sampler.step(model_output) 

    return sampler.get_last_sample()


def decode_latent(decoder, latent):
    img = checkpoint.checkpoint(decoder.decode, latent.unsqueeze(0) / decoder.config.scaling_factor,  use_reentrant=False).sample
    return img

def to_img(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img[0]

def compute_probability_regularization(noise_vectors, eta, opt_time, subsample, shuffled_times = 100):
    
    
    # squential subsampling
    if eta > 0:
        noise_vectors_flat = noise_vectors[:(opt_time + 1)].flatten()
    else:
        noise_vectors_flat = noise_vectors[-1].flatten()
        
    dim = noise_vectors_flat.shape[0]

    # use for computing the probability regularization
    subsample_dim = round(4 ** subsample)
    subsample_num = dim // subsample_dim
        
    noise_vectors_seq = noise_vectors_flat.view(subsample_num, subsample_dim)

    seq_mean = noise_vectors_seq.mean(dim = 0)
    noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
    seq_cov = noise_vectors_seq.T @ noise_vectors_seq
    seq_var = seq_cov.diag()
    
    # compute the probability of the noise
    seq_mean_M = torch.norm(seq_mean)
    seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
    
    seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
    seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
    seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
    seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
    seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
    seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
    seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)

    shuffled_mean_prob_list = []
    shuffled_cov_prob_list = [] 
    
    shuffled_mean_log_prob_list = []
    shuffled_cov_log_prob_list = [] 
    
    shuffled_mean_M_list = []
    shuffled_cov_M_list = []

    for _ in range(shuffled_times):
        noise_vectors_flat_shuffled = noise_vectors_flat[torch.randperm(dim)]   
        noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
        
        shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
        noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
        shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
        shuffled_var = shuffled_cov.diag()
        
        # compute the probability of the noise
        shuffled_mean_M = torch.norm(shuffled_mean)
        shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
        

        shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
        shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
        shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
        shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
        
        shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
        shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
        shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
        
        
        shuffled_mean_prob_list.append(shuffled_mean_prob.item())
        shuffled_cov_prob_list.append(shuffled_cov_prob.item())
        
        shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
        shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
        
        shuffled_mean_M_list.append(shuffled_mean_M.item())
        shuffled_cov_M_list.append(shuffled_cov_M.item())
        
    reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)
    
    return reg_loss

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: DNO_SDXL_promptset):
    
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
                
                image = run_DNO(opt)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_DNO(opt: DNO_SDXL_promptset):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda")
        
    # freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    # disable safety checker
    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # set the number of steps
    pipe.scheduler.set_timesteps(opt.num_inference_steps)
    unet = pipe.unet

    # load the loss function, which is negative of the reward fucntion
    loss_fn = RFUNCTIONS[opt.reward_type](inference_dtype = torch.float32, device = "cuda")

    torch.manual_seed(img_seed)
    noise_vectors = torch.randn(opt.num_inference_steps + 1, 4, 128, 128, device = "cuda")
    noise_vectors.requires_grad_(True)
    optimize_groups = [{"params":noise_vectors, "lr":opt.lr}]
    optimizer = torch.optim.AdamW(optimize_groups)
    
    (prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,    
        ) = pipe.encode_prompt(
                        prompt = prompt,
                        device = "cuda"
                    )
        
    
    # Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
            (1024, 1024),
            (0, 0),
            (1024, 1024),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    negative_add_time_ids = add_time_ids
    
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to("cuda")
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0).to("cuda")
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to("cuda")

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    

    # start optimization, opt fpr using fp16 mixed precision
    use_amp = False if opt.precision == "fp32" else True
    grad_scaler = GradScaler(enabled=use_amp, init_scale = 8192)
    amp_dtype = torch.bfloat16 if opt.precision == "bf16" else torch.float16
    
    

    for i in range(opt.opt_steps):
        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            ddim_sampler = SequentialDDIM(timesteps = opt.num_inference_steps,
                                            scheduler = pipe.scheduler, 
                                            eta = opt.eta, 
                                            cfg_scale = opt.guidance_scale, 
                                            device = "cuda",
                                            opt_timesteps = opt.opt_time)

            sample = sequential_sampling(pipe, unet, ddim_sampler, prompt_embeds = prompt_embeds,added_cond_kwargs = added_cond_kwargs, noise_vectors = noise_vectors)
            sample = decode_latent(pipe.vae, sample)
            

            losses = loss_fn(sample, [prompt] * sample.shape[0])
            loss = losses.mean()

            reward = -loss.item()
            
            if opt.gamma > 0:
                reg_loss = compute_probability_regularization(noise_vectors, opt.eta, opt.opt_time, opt.subsample)
                loss = loss + opt.gamma * reg_loss

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            
    
        img = to_img(sample)
        img = Image.fromarray(img)
    im = img    
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()
    
    