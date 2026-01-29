import os
import torch
import random


# data generating helper functions
def utility(X, device='cpu', maximize=True):
    """Given X, output corresponding utility (i.e., the latent function)"""
    # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
    weighted_X = X * torch.sqrt(torch.arange(X.size(-1), dtype=torch.float, device=device) + 1)
    y = torch.sum(weighted_X, dim=-1)
    if maximize:
        result = y
    else:
        result = -y
    return result.to(device)

def optimum(dims, device='cpu'):
    return utility(torch.tensor([[1] * dims]).to(device),device=device).item()