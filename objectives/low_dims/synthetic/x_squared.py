import os
import torch
import random


# data generating helper functions
def utility(X, device='cpu', maximize=True):
    """Given X, output corresponding utility (i.e., the latent function)"""
    # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
    
    y = X.norm()**2
    if maximize:
        result = -y
    else:
        result = y
    return y.to(device)

def optimum(dims):
    return utility(torch.tensor([[0] * dims])).item()