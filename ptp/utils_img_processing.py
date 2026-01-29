import torch
import torch.nn.functional as F
from utils.general_utils import set_all_seeds
import sys
# sys.path.append('../DenseMatching')
from utils_data.geometric_transformation_sampling.synthetic_warps_sampling import AffHomoTPSTransfo, AddElasticTransformsV2_det, AddGridDistortion
from utils_flow.pixel_wise_mapping import warp
import numpy as np

def center_pad(inp: torch.Tensor, size: tuple, kwargs: dict):
    if not isinstance(size, tuple):
        size = (size, size)
    target_h, target_w = size
    ndim = inp.ndim
    if ndim == 3:
        B, HW, T = inp.shape
        H = W = int(HW ** 0.5)
        inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)
    else:
        B, T, H, W = inp.shape
    if H < target_h:
        pad_y_total = target_h - H
        pad_y_1 = pad_y_total // 2
        pad_y_2 = pad_y_total - pad_y_1
    else:
        pad_y_1, pad_y_2 = 0, 0

    if W < target_w:
        pad_x_total = target_w - W
        pad_x_1 = pad_x_total // 2
        pad_x_2 = pad_x_total - pad_x_1
    else:
        pad_x_1, pad_x_2 = 0, 0
    padding_tuple = (pad_x_1, pad_x_2, pad_y_1, pad_y_2)
    if kwargs['mode'] == 'constant':
        inp_padded = F.pad(inp, padding_tuple, mode=kwargs['mode'], value=kwargs['value'])
    else:
        inp_padded = F.pad(inp, padding_tuple, mode=kwargs['mode'])
    
    if ndim == 3:
        inp_padded = inp_padded.permute(0, 2, 3, 1).reshape(B, target_h * target_w, T)
    return inp_padded

def get_center_crop_coords(height, width, crop_height, crop_width):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(inp, size):
    if not isinstance(size, tuple):
        size = (size, size)
    target_h, target_w = size
    ndim = inp.ndim
    if ndim == 3:
        B, HW, T = inp.shape
        H = W = int(HW ** 0.5)
        inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)
    else:
        B, T, H, W = inp.shape
        
    if H < target_h or W < target_w:
        raise ValueError(
            f'Requested crop size ({target_h}, {target_w}) is '
            f'larger than the image size ({H}, {W})'
        )
    x1, y1, x2, y2 = get_center_crop_coords(H, W, target_h, target_w)
    inp_cropped = inp[:,:,y1:y2, x1:x2]
    if ndim == 3:
        inp_cropped = inp_cropped.permute(0, 2, 3, 1).reshape(B, target_h * target_w, T)
    return inp_cropped


class WarpingTransform:
    def __init__(self, size_output_flow, geometric_model="afftps", _t=0.25, _s=0.5, _alpha=np.pi / 12,
                 _t_tps_for_afftps=None, _t_hom=0.4, _t_tps=0.4, tps_grid_size=3, tps_reg_factor=0,
                 transformation_types=None, _horizontal_flip=False, use_cuda=True,
                 nbr_perturbations=5, elastic_parameters=None, sigma_mask=7, device=None, seed=42, use_elastic=True, blend=False):
        geo_trans = AffHomoTPSTransfo(size_output_flow,_t,_s,_alpha,_t_tps_for_afftps,
                                      _t_hom, _t_tps, tps_grid_size, tps_reg_factor,
                                      transformation_types,_horizontal_flip,use_cuda)
        elas_trans = AddElasticTransformsV2_det(size_output_flow,nbr_perturbations,
                                              elastic_parameters,sigma_mask, device)
        self.geo_trans = geo_trans
        self.elas_trans = elas_trans
        self.geometric_model=geometric_model
        self.seed=seed
        self.device=device
        self.use_elastic=use_elastic
        self.blend=blend
    def __call__(self, feats, theta, training=True, flow=None, *args, **kwargs):
        """_summary_

        Args:
            feats (_type_): input feature to be warped [B, C, H, W]
            training (bool, optional): _description_. Defaults to True.
            flow (_type_, optional): geometric mapping. Defaults to None.
        """
        min_b = {'source_image':feats}
        set_all_seeds(self.seed)
        flow = self.geo_trans(theta=theta, geometric_model=self.geometric_model)
        if self.use_elastic:
            result_flow = self.elas_trans(mini_batch=min_b, flow=flow, seed=self.seed)
        else:
            result_flow = flow
        warped_feat = feats.clone()
        warp_mask = []
        # print("img proc ------", feats.shape[0])
        for i in range(feats.shape[0]):
            warped_feat_, warp_mask_ = warp(feats[i:i+1].float(),result_flow.to(self.device),return_mask=True)
            warp_mask.append(warp_mask_)
            warped_feat[i:i+1] = warped_feat_
        warped_mask = torch.stack(warp_mask)
        if self.blend:
            # print(warped_mask.shape, warped_feat.shape,feats.shape)
            # warped_feat = warped_mask.float() * warped_feat + (1-warped_mask.float()) * feats
            # warped_mask is a torch.bool tensor
            # warped_feat = torch.where(warped_mask, warped_feat, feats)
            lam = 0.9
            warped_feat = lam  * warped_feat + (1-lam) * feats
        return warped_feat.to(feats.dtype), warped_mask