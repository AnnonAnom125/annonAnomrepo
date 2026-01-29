import importlib
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from acfs.taf import TAFAcquisition
from acfs.twostep import TwoStepLookaheadPairwise
from acfs.twostep_taf import TwoStepLookahead
from acfs.dbs import DynamicBalancedSubspace
import torch
import math

def get_acf(acfs: str, opt: dict):
    """
    Dynamically load acquisition function module.

    Args:
        acfs (List[str]): name of acf algos, e.g. "qEI"
        opt (Dict): parameters

    Returns:
        function: list of acf functions 
    """
    module = []
    for acf_name in acfs:
        if acf_name == "EUBO":
            module_name = AnalyticExpectedUtilityOfBestOption
        elif acf_name == "qEI" or acf_name == "manifold":
            module_name = qLogNoisyExpectedImprovement
        elif acf_name == "TAFR-qEI":
            module_name = TAFAcquisition
        elif acf_name == "2s-qEI":
            module_name = TwoStepLookaheadPairwise
            # module_name = TwoStepLookahead(model, num_fantasies=opt.num_fantasies, use_taf=False, ref_points=opt.ref_points)
        elif acf_name == "2s-TAF-qEI":
            module_name = TwoStepLookahead
        elif acf_name == "UCB":
            module_name = qUpperConfidenceBound
        elif acf_name == "manifold-lookahead":
            module_name = [qLogNoisyExpectedImprovement, TwoStepLookaheadPairwise]
        elif acf_name == "manifold-dbs":
            module_name = [qLogNoisyExpectedImprovement, DynamicBalancedSubspace]
        elif acf_name == "rand":
            module_name = None
        else:
            raise ValueError(f"Unknown mode: {acf_name}")

        module.append(module_name)
    # Import module dynamically
    # module = importlib.import_module(module_name)

    # assume the function is named "f"
    return module

def construct_search_manifold(x_best, x_ei1, x_ei2, num_samples=9, lim=[-1.0,1.0]):
    
    # u and v define the D direction vectors from the best point to EI suggestions
    u = x_ei1 - x_best
    v = x_ei2 - x_best
    
    # Generate a grid of coefficients. 
    #  -1 to 1 is correct.
    side_len = int(math.sqrt(num_samples))
    ticks = torch.linspace(-1.0, 1.0, side_len).to(x_best.device)
    grid_alpha, grid_beta = torch.meshgrid(ticks, ticks, indexing='ij')
    
    # Reshape for broadcasting: (num_samples, 1)
    alpha = grid_alpha.reshape(-1, 1)
    beta = grid_beta.reshape(-1, 1)
    
    # Calculate points: x_new = x_best + alpha*(x_ei1-x_best) + beta*(x_ei2-x_best)
    # This spans the 20D space based on the directions identified by EI
    manifold_samples = x_best + (alpha * u) + (beta * v)
    
    # Clamp to your specific limits [-1, 1]
    return torch.clamp(manifold_samples, lim[0], lim[1])

