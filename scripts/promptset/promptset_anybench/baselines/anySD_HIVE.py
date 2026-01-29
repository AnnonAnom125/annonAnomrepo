'''
This is a sample script to show how anybench can be used:

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH='./' python3 anybench/eval/anybench_gen_eval.py
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..","..")))
current_dir = os.path.dirname(os.path.abspath(__file__))
import json
import pyrallis
from tqdm import tqdm
import PIL
import torch
from torchvision.transforms import transforms
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from configs import *
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageOps
from torch import autocast
import importlib
import einops
import math
from utils.anysd.src.model import AnySDPipeline, choose_expert
from utils.anysd.train.valid_log import download_image
from utils.anysd.src.utils import choose_book, get_experts_dir

def create_folder_path(opt):
    if opt.obj_name in opt.image_models:
        pp = os.path.join(opt.output_path,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: AnySD_HIVE_promptset):
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
                inp_path = os.path.join(opt.output_path , "input_image")
                os.makedirs(inp_path,exist_ok=True)
                
                opt.prompts = prompt_ed_lst["input"]
                prompt = prompt_ed_lst["input"]
                opt.edit = prompt_ed_lst["edit"]
                opt.edit_object = prompt_ed_lst["edit object"]
                opt.edit_type = prompt_ed_lst["edit_type"]
                
                pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16, use_safetensors=True,).to("cuda")
                gen = torch.Generator().manual_seed(img_seed)
                inp_image = pipe(prompt, num_inference_steps=opt.num_inference_steps, generator=gen)['images'][0]
                inp_image.save(os.path.join(inp_path, f"{prompt}_{img_seed}.png") )
                opt.image_file = os.path.join(inp_path, f"{prompt}_{img_seed}.png") 
                
                print(" ")
                print(f"Prompt: {opt.prompts}")
                print(" ")
                
                image = run_anySD_HIVE(opt)
                # image is in the order of acf_algos x trials
                for i in range(len(image)):
                    save_path = os.path.join(op_path, f"{prompt}_{counter_id:06d}.png") 
                    image[i].save(save_path)
                    print(" ")
                    counter_id += 1
    

def run_anySD_HIVE(opt: AnySD_HIVE_promptset):
    model_name = opt.model_name
    
    if model_name != 'anysd':
        if model_name == 'hivec':
            model = HIVEc(opt.model_path)
        elif model_name == 'hivew':
            model = HIVEw(opt.model_path)
        

        im = model.edit(image_url=opt.image_file,
                        prompt=opt.prompts,
                        save_path=None)
    else:

        expert_file_path = get_experts_dir(repo_id="WeiChow/AnySD")
        book_dim, book = choose_book('all')
        task_embs_checkpoints = expert_file_path + "task_embs.bin"
        adapter_checkpoints = {
            "global": expert_file_path + "global.bin",
            "viewpoint": expert_file_path + "viewpoint.bin",
            "visual_bbox": expert_file_path + "visual_bbox.bin",
            "visual_depth": expert_file_path + "visual_dep.bin",
            "visual_material_transfer": expert_file_path + "visual_mat.bin",
            "visual_reference": expert_file_path + "visual_ref.bin",
            "visual_scribble": expert_file_path + "visual_scr.bin",
            "visual_segment": expert_file_path + "visual_seg.bin",
            "visual_sketch": expert_file_path + "visual_ske.bin",
        }

        pipeline = AnySDPipeline(adapters_list=adapter_checkpoints, task_embs_checkpoints=task_embs_checkpoints)

        
        mode = choose_expert(mode=opt.edit_type)
        if mode == 'general':
            images = pipeline(
                prompt=opt.edit,
                original_image=download_image(opt.image_file),
                guidance_scale=3,
                num_inference_steps=100,
                original_image_guidance_scale=3,
                adapter_name="general",
            )[0]
        else:
            im = pipeline(
                prompt=opt.edit,
                reference_image=None,
                original_image=download_image(opt.image_file),
                guidance_scale=1.5,
                num_inference_steps=100,
                original_image_guidance_scale=2,
                reference_image_guidance_scale=0.8,
                adapter_name=mode,
                e_code=book[opt.edit_type],
            )[0]

    result_final = [im]*opt.num_trials
    return result_final
    
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    import k_diffusion as K
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

class HIVEw():
    def __init__(self, model_path="./anybench/checkpoints/hive/hive_rw.ckpt"):
        import k_diffusion as K
        self.resolution = 512
        self.steps = 100
        self.seed = 100

        config_path = os.path.join(current_dir, "hive/generate.yaml")
        config = OmegaConf.load(config_path)

        self.model = load_model_from_config(config, model_path, None)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    @torch.inference_mode()
    def edit(self, image_url, prompt):
        import k_diffusion as K
        if type(image_url) is str:
            input_image = image_url.convert("RGB")
        else:
            input_image = PIL.Image.open(image_url).convert("RGB")

        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([prompt])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(self.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        return edited_image

class HIVEc(HIVEw):
    def __init__(self, model_path="./anybench/checkpoints/hive/hive_rw_condition.ckpt"):
        super().__init__(model_path=model_path)

    @torch.inference_mode()
    def edit(self, image_url, prompt):
        if type(image_url) is str:
            input_image = Image.open(image_url).convert("RGB")
        else:
            input_image = image_url.convert("RGB")

        import k_diffusion as K
        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            edit = prompt + ', ' + f'image quality is five out of five'
            cond["c_crossattn"] = [self.model.get_learned_conditioning([edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(self.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        return edited_image

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
