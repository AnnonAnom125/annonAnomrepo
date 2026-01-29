import os
import numpy as np
import torch
import random
from itertools import combinations

from models.multiwise_gp import MultiwiseGP
from models.pairwise_gp_new import PairwiseGP
# from botorch.models.pairwise_gp import PairwiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from scipy.stats import kendalltau




def generate_batch_initial_conditions(
    bounds: torch.Tensor,
    num_restarts: int,
    q: int,
    dtype=torch.float64,
    device='cpu',
    existing_points: torch.Tensor = None
) -> torch.Tensor:
    """
    Generate initial conditions for optimize_acqf to avoid BoTorch's internal batch initialization.

    Args:
        bounds: Tensor of shape (2, d), lower and upper bounds.
        num_restarts: Number of restart points for the optimizer.
        q: Batch size (number of points per candidate batch).
        dtype: torch dtype.
        device: torch device.
        existing_points: Optional tensor of shape (n_points, d) to sample from.
    
    Returns:
        Tensor of shape (num_restarts, q, d) suitable for `batch_initial_conditions`.
    """
    d = bounds.shape[-1]
    batch_init = torch.empty(num_restarts, q, d, dtype=dtype, device=device)

    if existing_points is not None:
        n_existing = existing_points.shape[0]
        for i in range(num_restarts):
            idx = torch.randint(0, n_existing, (q,))
            batch_init[i] = existing_points[idx].to(dtype=dtype, device=device)
    else:
        lower, upper = bounds
        batch_init.uniform_(0, 1).to(device)
        batch_init = lower + (upper - lower) * batch_init

    return batch_init


def init_and_fit_model(X, comp, likelihood, device='cpu', ch=None, type_likelihood=False, multi=False):
    """Model fitting helper function"""
    if multi:
        model = MultiwiseGP(
        X,
        comp,
        ch,
        input_transform=Normalize(d=X.shape[-1]),
        )
        mll = likelihood(model.likelihood, model)
    else:
        model = PairwiseGP(
            X,
            comp,
            input_transform=Normalize(d=X.shape[-1]),
            type_likelihood=type_likelihood,
        )
        mll = likelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model


# Kendall-Tau rank correlation
def eval_kt_cor(model, test_X, test_y, device='cpu'):
    pred_y = model.posterior(test_X).mean.squeeze().detach().cpu().numpy()
    return kendalltau(pred_y, test_y.cpu()).correlation


# def create_population_models(
#         opt, 
#         train_X,
#         train_y,
#         train_comp,
#         train_ch = None,
#         likelihood = None
#         ):
#     old_models = []
#     old_mll = []
#     for i in range(0,opt.num_old_models):
#         if opt.multi:
#             old_model = MultiwiseGP(
#             train_X,
#             train_comp,
#             train_ch,
#             input_transform=Normalize(d=train_X.shape[-1]),
#             )
#             old_mll = likelihood(old_model.likelihood, old_model)
#         else:
#             old_model = PairwiseGP(
#                 train_X,
#                 train_comp,
#                 input_transform=Normalize(d=train_X.shape[-1]),
#                 type_likelihood=True,
#             )
#             old_mll = likelihood(old_model.likelihood, old_model)
#         old_mll = fit_gpytorch_mll(old_mll)
#     return old_models, old_mll
