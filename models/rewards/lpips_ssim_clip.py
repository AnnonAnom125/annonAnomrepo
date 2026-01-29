import torch
import clip
from PIL import Image
import pyiqa
import torchvision.transforms as tf

class Selector():
    
    def __init__(self, device):
        self.device = device
        self.clip_model, self.preprocess_clip = clip.load("ViT-B/16", device)
        self.metric_ssim = pyiqa.create_metric('ssim', device=device)
        self.metric_lpips = pyiqa.create_metric('lpips', as_loss=False, device=device)
        self.preprocess_pil = tf.PILToTensor()
    def score(self, tar_img_path, edit_img_path):
        if isinstance(edit_img_path, list):
            result = []
            for one_img_path, one_tar_img_path in zip(edit_img_path,tar_img_path):
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    if isinstance(one_img_path, str):
                        edit_image_pil = Image.open(one_img_path).convert('RGB')
                    elif isinstance(one_img_path, Image.Image):
                        edit_image_pil = one_img_path
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    if isinstance(one_img_path, str):
                        tar_image_pil = Image.open(one_tar_img_path).convert('RGB')
                    elif isinstance(one_tar_img_path, Image.Image):
                        tar_image_pil = one_tar_img_path
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    
                    target_clip_input = self.preprocess_clip(tar_image_pil).unsqueeze(0).to(self.device, non_blocking=True)
                    target_emb = self.clip_model.encode_image(target_clip_input)
                    target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)
                    
                    img_clip_input = self.preprocess_clip(edit_image_pil).unsqueeze(0).to(self.device, non_blocking=True)
                    img_emb = self.clip_model.encode_image(img_clip_input)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    
                    score_clip = float((target_emb @ img_emb.T).item())
                    score_ssim = float(self.metric_ssim(edit_image_pil, tar_image_pil).item())
                    score_lpips_dist = float(self.metric_lpips(edit_image_pil, tar_image_pil).item())
                    score_lpips = max(0, 1 - score_lpips_dist)
                    combined_score = (score_ssim + score_lpips + score_clip) / 3
                result.append(combined_score)    
            return result
        elif isinstance(edit_img_path, str):
            with torch.no_grad():
                # Process the image
                edit_image_pil = Image.open(one_img_path).convert('RGB')
                tar_image_pil = one_tar_img_path
                
                target_clip_input = self.preprocess_clip(tar_image_pil).unsqueeze(0).to(self.device, non_blocking=True)
                target_emb = self.clip_model.encode_image(target_clip_input)
                target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)
                
                img_clip_input = self.preprocess_clip(edit_image_pil).unsqueeze(0).to(self.device, non_blocking=True)
                img_emb = self.clip_model.encode_image(img_clip_input)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                
                score_clip = float((target_emb @ img_emb.T).item())
                score_ssim = float(self.metric_ssim(edit_image_pil, tar_image_pil).item())
                score_lpips_dist = float(self.metric_lpips(edit_image_pil, tar_image_pil).item())
                score_lpips = max(0, 1 - score_lpips_dist)
                combined_score = (score_ssim + score_lpips + score_clip) / 3
            return [combined_score]
        
                        

