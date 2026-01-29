import torch
import clip
from PIL import Image
import pyiqa
import torchvision.transforms as tf

class Selector():
    
    def __init__(self, metric, device):
        self.device = device
        self.metric = pyiqa.create_metric(metric, device=device)
    
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
                    
                    if self.metric in ["liqe","musiq","topiq"]:
                        score_metric = float(self.metric(edit_image_pil).item())
                    else:
                        score_metric = float(self.metric(edit_image_pil, tar_image_pil).item())
                    
                result.append(score_metric)    
            return result
        elif isinstance(edit_img_path, str):
            with torch.no_grad():
                # Process the image
                edit_image_pil = Image.open(one_img_path).convert('RGB')
                tar_image_pil = one_tar_img_path
                if self.metric in ["liqe","musiq","topiq"]:
                        score_metric = float(self.metric(edit_image_pil).item())
                else:
                    score_metric = float(self.metric(edit_image_pil, tar_image_pil).item())
            return [score_metric]
        
                        

