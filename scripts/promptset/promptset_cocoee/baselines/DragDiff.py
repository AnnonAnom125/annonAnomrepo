import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..","..")))
import json
import pyrallis
import numpy as np
import torch
import random
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from configs import *
from PIL import Image
from utils.dragdiff.drag_pipeline import DragPipeline
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from einops import rearrange
from types import SimpleNamespace
import torch.nn.functional as F
from copy import deepcopy
from utils.dragdiff.drag_utils import drag_diffusion_update
from utils.dragdiff.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.dragdiff.lora_utils import train_lora

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: DragDiff_promptset):
    
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
                original_image = np.array(Image.open(image_path))
                mask = np.array(Image.open(mask_path).convert('L'))
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
                
                image = run_DragDiff(opt, original_image, mask)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
                
def run_DragDiff(opt: DragDiff_promptset, source_image, mask):
    
    prompt = opt.prompts 
    img_seed = opt.img_seed
    opt.output_path = create_folder_path(opt)
    os.makedirs(opt.output_path,exist_ok=True)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(img_seed)
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()
    
    # set vae
    if opt.vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(opt.vae_path).to(model.vae.device, model.vae.dtype)

    # initialize parameters
    seed_everything(img_seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = opt.selected_points
    args.n_inference_step = opt.num_inference_steps
    args.n_actual_inference_step = round(opt.inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [opt.unet_feature_idx]

    args.r_m = opt.r_m
    args.r_p = opt.r_p
    args.lam = opt.lam

    args.lr = opt.latent_lr
    args.n_pix_step = opt.n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)
    
    lora_path = os.path.join(opt.output_path,opt.lora_str)
    train_lora(
        source_image,
        prompt,
        "runwayml/stable-diffusion-v1-5",
        opt.vae_path,
        lora_path,
        opt.lora_step,
        opt.lora_lr,
        opt.lora_batch_size,
        opt.lora_rank)

    opt.lora_path = lora_path
    # print(args)

    source_image = preprocess_image(source_image, device)
    # image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        model.unet.set_default_attn_processor()
    else:
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step)

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(opt.selected_points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    updated_init_code = drag_diffusion_update(model, init_code,
        None, t, handle_points, target_points, mask, args)

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=opt.start_step,
                                        start_layer=opt.start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # inference the synthesized image
    gen_image = model(
        prompt=args.prompt,
        batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
        )[1].unsqueeze(dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    im = Image.fromarray(out_image)
    
    result_final = [im]*opt.num_trials
    return result_final

if __name__ == "__main__":
    start()

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image 