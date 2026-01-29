import importlib
import os
import numpy as np
import torch
import random
import itertools
from itertools import combinations
from torchvision import transforms
from PIL import Image
from botorch.utils.sampling import draw_sobol_samples

def get_objective(obj_name: str, mode: str = "low"):
    """
    Dynamically load objective function module.

    Args:
        obj_name (str): name of objective file, e.g. "xx"
        mode (str): "low" or "high"

    Returns:
        function: objective function f
    """
    if mode == "low-synthetic":
        module_name = f"objectives.low_dims.synthetic.{obj_name}"
    elif mode == "high-synthetic":
        module_name = f"objectives.high_dims.synthetic.{obj_name}"
    elif mode == "low-image":
        module_name = f"objectives.low_dims.ImageGen.{obj_name}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Import module dynamically
    module = importlib.import_module(module_name)

    # assume the function is named "f"
    return module.utility, module.optimum, module_name

def get_objective_subspace(obj_name: str, mode: str = "low"):
    """
    Dynamically load objective function module.

    Args:
        obj_name (str): name of objective file, e.g. "xx"
        mode (str): "low" or "high"

    Returns:
        function: objective function f
    """
    if mode == "low-synthetic":
        module_name = f"objectives.low_dims.synthetic.{obj_name}"
    elif mode == "high-synthetic":
        module_name = f"objectives.high_dims.synthetic.{obj_name}"
    elif mode == "low-image":
        module_name = f"objectives.low_dims.ImageGen.{obj_name}_subspace"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Import module dynamically
    module = importlib.import_module(module_name)

    # assume the function is named "f"
    return module.utility, module.optimum, module_name

def generate_data(utility, n, dim=2, device='cpu', opt=None, kwargs=None):
    """Generate data X and y"""
    type = kwargs['type']
    if type == 'uniform':
        X = torch.rand(n, dim, dtype=torch.float64, device=device) * (opt.lim[1] - opt.lim[0]) + opt.lim[0]
    else:
        bounds = torch.stack([opt.lim[0]*torch.ones(opt.dim), opt.lim[1]*torch.ones(opt.dim)]).to(device)
        X = draw_sobol_samples(bounds=bounds,n=n,q=1).squeeze(1)
    y = utility(X,device=device)
    return X, y

def generate_data_image(utility, n, dim=2, device='cpu', opt=None, kwargs=None):
    """Generate data X and y"""
    opt = kwargs["opt"]
    type = kwargs['type']
    if type == 'uniform':
        X = torch.rand(n, dim, dtype=torch.float64, device=device) * (opt.lim[1] - opt.lim[0]) + opt.lim[0]
    else:
        bounds = torch.stack([opt.lim[0]*torch.ones(opt.dim), opt.lim[1]*torch.ones(opt.dim)]).to(device)
        X = draw_sobol_samples(bounds=bounds,n=n,q=1).squeeze(1)
    y = utility(X,opt, device=device)
    
    return X, y

def generate_comparisons_pair(y, n_comp, noise=0.1, T=None, replace=False, device='cpu', kwargs=None):
    """Create pairwise comparisons with noise"""
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = torch.tensor(all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ],device=device)
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + torch.randn(len(comp_pairs),device=device) * noise
    c1 = y[comp_pairs[:, 1]] + torch.randn(len(comp_pairs),device=device) * noise
    reverse_comp = (c0 < c1)#.numpy()
    comp_pairs[reverse_comp, :] = torch.flip(comp_pairs[reverse_comp, :], [1])
    comp_pairs = torch.tensor(comp_pairs, device=device).long()
    return comp_pairs, None

def generate_comparisons_multi(y, n_comp, T=3, noise=0.1, replace=False, device='cpu', kwargs=None):
    """
    Generate multiwise comparisons with choices.
    """
    n = len(y)
    comparisons = []
    choices = []
    for _ in range(n_comp):
        if replace:
            cand = np.random.choice(n, T, replace=True)
        else:
            cand = np.random.choice(n, T, replace=False)
        noisy_utils = y[cand] + torch.randn(T, device=device) * noise
        winner = torch.argmax(noisy_utils)
        comparisons.append(cand)
        choices.append(winner)
    comparisons = torch.tensor(comparisons, device=device).long()  # (n_trials, T)
    choices = torch.tensor(choices, device=device).long()         # (n_trials,)
    return comparisons, choices

def generate_comparisons_image(y, n_comp, T=3, noise=0.1, replace=False, device='cpu', kwargs=None):
    """
    Generate multiwise comparison images.
    """
    obj = kwargs["obj"]
    obj = importlib.import_module(obj)
    title = kwargs['title']
    json_input = kwargs['json_input']
    orig_path = kwargs["orig_path"]
    opt = kwargs["opt"]
    target_img = kwargs.get("target_img", None)
    n = len(y)
    all_possible_pairs = list(itertools.combinations(range(n), T))
    if n_comp > len(all_possible_pairs):
        raise ValueError(f"Cannot pick {n_comp} unique pairs without replacement from only {len(all_possible_pairs)} possible pairs.")
    all_possible_pairs_np = np.array(all_possible_pairs)
    indices = np.random.choice(len(all_possible_pairs_np), n_comp, replace=replace)
    selected_pairs = all_possible_pairs_np[indices]
    comparisons = []
    choices = []
    for cand in selected_pairs:
        inp = [y[i] for i in cand]
        if not opt.non_human_score:
            idx = obj.record_choice(inp, title, json_input=json_input, orig_path=orig_path, target=target_img)
        else:
            idx = obj.non_human_choice(inp, opt.prompts, opt.score_metric, opt, target=target_img, device=device) 
        winner = cand[idx] 
        cc = [winner] + [c for i, c in enumerate(cand) if i != idx]
        # cc = [winner] + list(cand[:idx]) + list(cand[idx+1:])
        comparisons.append(cc)
        choices.append(winner)
    comparisons = torch.tensor(comparisons, device=device).long()  # (n_trials, T)
    choices = torch.tensor(choices, device=device).long()          # (n_trials,)
    # print(" ")
    # print("choice--",choices, comparisons)
    # print(" ")
    return comparisons, choices


def find_best_image(X, y, title=None, json_input=".json", orig_path=None, kwargs=None):
    obj = kwargs["obj"]
    opt = kwargs["opt"]
    target_img = kwargs.get("target_img", None)
    obj = importlib.import_module(obj)
    n = len(y)
    if not opt.non_human_score:
        winner = obj.record_choice(y,title,json_input=json_input,orig_path=orig_path, target=target_img)
    else:
        winner = obj.non_human_choice(y, opt.prompts, opt.score_metric, opt, target=target_img, device=opt.device) 
    return X[winner].unsqueeze(0), y[winner]


def make_new_data(utility, X, next_X, y, comps, q_comp, T=None, multi=False, choices=None, device='cpu', noise=None):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    next_X = next_X.to(X)
    next_y = utility(next_X, device=device)
    if multi:
        next_comps, next_choices = generate_comparisons_multi(next_y, n_comp=q_comp, T=T, noise=noise, device=device)
        comps = torch.cat([comps, next_comps + X.shape[-2]])
        chs = torch.cat([choices, next_choices.to(X.device)], dim=-1)
        X = torch.cat([X, next_X])
        y = torch.cat([y, next_y])
        return X, y, comps, chs
    else:
        next_comps, _ = generate_comparisons_pair(next_y, n_comp=q_comp, noise=noise, device=device)
        comps = torch.cat([comps, next_comps + X.shape[-2]])
        X = torch.cat([X, next_X])
        y = torch.cat([y, next_y])
        return X, y, comps, None


def make_new_data_image(utility, X, next_X, y, comps, q_comp, T=None, multi=False, choices=None, device='cpu', noise=None, kwargs=None):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    opt = kwargs["opt"]
    next_X = next_X.to(X)
    next_y = utility(next_X,opt, device=device)
    next_comps, next_choices = generate_comparisons_image(next_y, n_comp=q_comp, T=T, noise=noise, device=device, kwargs=kwargs)
    if opt.convert_to_pair:
        next_comps, next_choices = convert_t_way_to_pairwise(next_X, next_comps)
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    chs = torch.cat([choices, next_choices.to(X.device) + X.shape[-2]])
    X = torch.cat([X, next_X])
    if isinstance(y, list):
        y = y + next_y
    else:
        y = torch.cat([y, next_y])
    print(" ")
    print("make_new-----", comps, chs)
    print(" ")
    return X, y, comps, chs

def convert_t_way_to_pairwise(X, t_comp):
    comparisons = []
    choices = []
    
    # The chosen point is compared against every other point in the set
    for i in range(len(t_comp)):
        for idx in range(len(X)):
            if idx != t_comp[i,0]:
                comparisons.append([t_comp[i,0], idx])
                choices.append([t_comp[i,0]])
    return torch.tensor(comparisons, dtype=torch.long,device=t_comp.device), torch.tensor(choices, dtype=torch.long,device=t_comp.device)
