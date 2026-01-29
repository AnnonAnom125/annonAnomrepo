import os
import warnings
from itertools import combinations

import numpy as np
import torch
import random
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from pairwise_gp_new import PairwiseGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.transforms.input import Normalize
from scipy.stats import kendalltau
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt
import argparse
from taf import TAFAcquisition
from twostep import TwoStepLookaheadPairwise
from twostep_taf import TwoStepLookahead
from multiwise_gp import MultiwiseGP, MultiwiseLaplaceMarginalLogLikelihood

from configs import MultiBOConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,5,4,3,2,1,0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Multiwise/Pairwise BO Experiment")

    # Experiment setup
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--num_batches", type=int, default=30, help="Number of batches")

    # Problem setup
    parser.add_argument("--dim", type=int, default=4, help="Input dimension")
    parser.add_argument("--n", type=int, default=50, help="Number of datapoints")
    parser.add_argument("--m", type=int, default=100, help="Number of comparisons")

    # Acquisition parameters
    parser.add_argument("--num_restarts", type=int, default=3, help="Number of restarts for acquisition optimization")
    parser.add_argument("--raw_samples", type=int, default=512, help="Number of raw samples for acquisition optimization")
    parser.add_argument("--q", type=int, default=2, help="Number of points per query")
    parser.add_argument("--q_comp", type=int, default=1, help="Number of comparisons per query")

    # Noise / randomness
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise level")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--n_kendall", type=int, default=1000, help="Number of samples for Kendall's tau estimation")

    # Likelihood/modeling
    parser.add_argument("--logit", action="store_true", default=True, help="Use logit likelihood")
    parser.add_argument("--multi", action="store_true", default=False, help="Use multiwise comparisons")

    # Meta models/ TAF
    parser.add_argument("--num_old_models", type=int, default=2, help="Number of src models")
    parser.add_argument("--n_src", type=int, default=2, help="Number of datapoints for each src model")
    parser.add_argument("--num_fantasies", type=int, default=2, help="Number of fantasies")
    parser.add_argument("--ref_points", default=None, help="Fantasy reference points")
    parser.add_argument("--rho", type=float, default=0.1, help="rho")
    parser.add_argument("--d1", type=float, default=2, help="d1")
    parser.add_argument("--d2", type=float, default=0.05, help="d2")
    parser.add_argument("--increment", action="store_true", default=False, help="Increment iteration count")

    
    # Algos and plotting
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["EUBO", "rand", "qEI", "2s-qEI", "TAFR-qEI", "2s-TAF-qEI"],
        help="List of algorithms to run"
    )
    parser.add_argument("--plotting", action="store_true", default=True, help="Enable plotting")

    return parser.parse_args()

def set_all_seeds(seed):
    """
    Sets the random seed for numpy, random, and pytorch (CPU and CUDA).
    Also sets environment variable for Python hash seed and CUDNN for deterministic behavior.
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # Set Python hash seed
    random.seed(seed) # Set Python's built-in random module seed
    np.random.seed(seed) # Set NumPy's random seed
    torch.manual_seed(seed) # Set PyTorch's CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set PyTorch's current GPU random seed
        torch.cuda.manual_seed_all(seed) # Set PyTorch's all GPUs random seed
    
    # For deterministic behavior with CUDA operations (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        batch_init.uniform_(0, 1)
        batch_init = lower + (upper - lower) * batch_init

    return batch_init

def create_source_models(opt):
    old_models = []
    old_mll = []
    for i in range(0,opt.num_old_models):
        train_X, train_y = generate_data(opt.n_src, dim=opt.dim)
        if opt.multi:
            train_comp, train_ch = generate_comparisons(train_y, opt.m, T=opt.T, noise=opt.noise, device=device)
            old_models.append(MultiwiseGP(
            train_X,
            train_comp,
            train_ch,
            input_transform=Normalize(d=train_X.shape[-1]),
            ))
            old_mll.append(MultiwiseLaplaceMarginalLogLikelihood(old_models[i].likelihood, old_models[i]))
        else:
            train_comp = generate_comparisons(train_y, opt.m, noise=opt.noise, device=device)
            old_models.append(PairwiseGP(
                train_X,
                train_comp,
                input_transform=Normalize(d=train_X.shape[-1]),
                type_likelihood=True,
            ))
            old_mll.append(PairwiseLaplaceMarginalLogLikelihood(old_models[i].likelihood, old_models[i]))
        old_mll[i] = fit_gpytorch_mll(old_mll[i])
    return old_models, old_mll

# data generating helper functions
def utility(X, device='cpu'):
    """Given X, output corresponding utility (i.e., the latent function)"""
    # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
    weighted_X = X * torch.sqrt(torch.arange(X.size(-1), dtype=torch.float, device=device) + 1)
    y = torch.sum(weighted_X, dim=-1)
    return y.to(device)


def generate_data(n, dim=2, device='cpu'):
    """Generate data X and y"""
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    X = torch.rand(n, dim, dtype=torch.float64, device=device)
    y = utility(X)
    return X, y


def generate_comparisons(y, n_comp, noise=0.1, replace=False, device='cpu'):
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

    return comp_pairs

def generate_comparisons_multi(y, n_trials, T=3, noise=0.1, replace=False, device='cpu'):
    """
    Generate multiwise comparisons with choices.
    """
    n = len(y)
    comparisons = []
    choices = []
    for _ in range(n_trials):
        if replace:
            cand = np.random.choice(n, T, replace=True)
        else:
            cand = np.random.choice(n, T, replace=False)
        noisy_utils = y[cand] + torch.randn(T) * noise
        winner = torch.argmax(noisy_utils)
        comparisons.append(cand)
        choices.append(winner)
    comparisons = torch.tensor(comparisons).long()  # (n_trials, T)
    choices = torch.tensor(choices).long()          # (n_trials,)
    return comparisons, choices

def init_and_fit_model(X, comp, device='cpu', ch=None, multi=False):
    """Model fitting helper function"""
    if multi:
        model = MultiwiseGP(
        X,
        comp,
        ch,
        input_transform=Normalize(d=X.shape[-1]),
        )
        mll = MultiwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    else:
        model = PairwiseGP(
            X,
            comp,
            input_transform=Normalize(d=X.shape[-1]),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model


def make_new_data(X, next_X, comps, q_comp, T=None, multi=False, choices=None, device='cpu', noise=None):
    """Given X and next_X,
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    next_X = next_X.to(X)
    next_y = utility(next_X, device=device)
    if multi:
        next_comps, next_choices = generate_comparisons_multi(train_y, n_trials=q_comp, T=T, noise=noise, device=device)
        comps = torch.cat([comps, next_comps + X.shape[-2]])
        chs = torch.cat([choices, next_choices.to(X.device)], dim=-1)
        X = torch.cat([X, next_X])
        return X, comps, chs
    else:
        next_comps = generate_comparisons(next_y, n_comp=q_comp, noise=noise, device=device)
        comps = torch.cat([comps, next_comps + X.shape[-2]])
        X = torch.cat([X, next_X])
        return X, comps

# Kendall-Tau rank correlation
def eval_kt_cor(model, test_X, test_y, device='cpu'):
    pred_y = model.posterior(test_X).mean.squeeze().detach().numpy()
    return kendalltau(pred_y, test_y).correlation


opt = get_args()

set_all_seeds(opt.seed)

train_X, train_y = generate_data(opt.n, dim=opt.dim)


if opt.logit:
    opt.type_likelihood=True
else:
    opt.type_likelihood=False


opt.algos = opt.algos.append("rand")

if opt.multi:
    train_comp, train_ch = generate_comparisons_multi(train_y, opt.m, T=opt.T, noise=opt.noise, device=device)
    model = MultiwiseGP(
    train_X,
    train_comp,
    train_ch,
    input_transform=Normalize(d=train_X.shape[-1]),
    )
    mll = MultiwiseLaplaceMarginalLogLikelihood(model.likelihood, model)

else:
    train_comp = generate_comparisons(train_y, opt.m, noise=opt.noise, device=device)
    model = PairwiseGP(
        train_X,
        train_comp,
        input_transform=Normalize(d=train_X.shape[-1]),
        type_likelihood = opt.type_likelihood,
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
mll = fit_gpytorch_mll(mll)


test_X, test_y = generate_data(opt.n_kendall, dim=opt.dim)
kt_correlation = eval_kt_cor(model, test_X, test_y)

print(f"Test Kendall-Tau rank correlation: {kt_correlation:.4f}")


# initial evals
best_vals = {}  # best observed values
for algo in opt.algos:
    best_vals[algo] = []



# average over multiple trials
for i in range(opt.num_trials):
    torch.manual_seed(i)
    np.random.seed(i)
    data = {}
    models = {}

    # Create initial data
    init_X, init_y = generate_data(opt.q, dim=opt.dim)
    init_X_new, _ = generate_data(opt.num_restarts * opt.q, dim=opt.dim)
    # print(init_X.shape,init_y.shape)
    if opt.multi:
        comparisons, choices = generate_comparisons_multi(init_y, opt.q_comp, T=opt.T, noise=opt.noise,device=device)
        assert opt.q == opt.T
    else:
        comparisons = generate_comparisons(init_y, opt.q_comp, noise=opt.noise,device=device)
    # X are within the unit cube
    bounds = torch.stack([torch.zeros(opt.dim), torch.ones(opt.dim)])

    batch_initial_conditions = generate_batch_initial_conditions(
    bounds=bounds,
    num_restarts=opt.num_restarts,
    q=opt.q,
    existing_points=init_X_new,  # optional
    device=device,
)


    for algo in opt.algos:

        if algo == "EUBO":
            acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        elif algo == "qEI":
            acq_func = qExpectedImprovement(model=model, best_f = init_y.max())
        elif algo == "TAFR-qEI":
            opt.increment = True
            old_models, _ = create_source_models(opt)
            acq_func = TAFAcquisition(
                                        model_target=model,
                                        source_models=old_models,
                                        rho=opt.rho,
                                        d1=opt.d1,
                                        d2=opt.d2
                                    )
        elif algo =="2s-qEI":
            if opt.ref_points is None:
                opt.ref_points = train_X[:10]
            else:
                opt.ref_points = torch.load(opt.ref_points,device=device)


            acq_func = TwoStepLookaheadPairwise(
                model=model,
                inner_acq_cls=qExpectedImprovement,
                num_fantasies=opt.num_fantasies,
                ref_points=opt.ref_points,  # optional
            )
            # acq_func = TwoStepLookahead(model, num_fantasies=opt.num_fantasies, use_taf=False, ref_points=opt.ref_points)
        
        elif algo == "2s-TAF-qEI":
            opt.increment = True
            old_models, _ = create_source_models(opt)
            if opt.ref_points is None:
                opt.ref_points = train_X[:10]
            else:
                opt.ref_points = torch.load(opt.ref_points,device=device)
            acq_func = TwoStepLookahead(
                                model,
                                num_fantasies=opt.num_fantasies,
                                use_taf=True,
                                source_models=old_models,
                                inner_acq_kwargs={"rho": opt.rho, "d1": opt.d1, "d2": opt.d2},
                                ref_points=opt.ref_points,  # optional
                            )
            

    
        best_vals[algo].append([])
        if opt.multi:
            data[algo] = (init_X, comparisons, choices)
            _, models[algo] = init_and_fit_model(init_X, comparisons, ch=choices, multi=True, device=device)
        else:
            data[algo] = (init_X, comparisons)
            _, models[algo] = init_and_fit_model(init_X, comparisons, device=device)

        best_next_y = utility(init_X).max().item()
        best_vals[algo][-1].append(best_next_y)

    # we make additional num_batches comparison queries after the initial observation
    for j in range(1, opt.num_batches + 1):
        for algo in opt.algos:
            model = models[algo]
            if algo != "rand":
                # create the acquisition function object
                
                # optimize and get new observation
                next_X, acq_val = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=opt.q,
                    num_restarts=opt.num_restarts,
                    raw_samples=opt.raw_samples,
                    batch_initial_conditions=batch_initial_conditions,
                )
                print(f"Trials={i},batches={j}")
                if opt.increment:
                    acq_func.increment_iteration()
            else:
                # randomly sample data
                next_X, _ = generate_data(opt.q, dim=opt.dim)

            # update data
            # refit models
            if opt.multi:
                X, comps, chs = data[algo]
                X, comps, chs = make_new_data(X, next_X, comps, opt.q_comp, choices=chs, multi=True, T=opt.T, device=device, noise=opt.noise)
                data[algo] = (X, comps, chs)
                _, models[algo] = init_and_fit_model(X, comps, ch=chs, device=device, multi=True)
            else:
                X, comps = data[algo]
                X, comps = make_new_data(X, next_X, comps, opt.q_comp, device=device, noise=opt.noise)
                data[algo] = (X, comps)
                _, models[algo] = init_and_fit_model(X, comps, device=device)

            # record the best observed values so far
            max_val = utility(X).max().item()
            best_vals[algo][-1].append(max_val)

if opt.plotting:
    plt.rcParams.update({"font.size": 14})

    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(y.shape[0])


    # the utility function is maximized at the full vector of 1
    optimal_val = utility(torch.tensor([[1] * opt.dim])).item()
    iters = list(range(opt.num_batches + 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # plot the optimal value
    ax.plot(
        iters,
        [optimal_val] * len(iters),
        label="Optimal Function Value",
        color="black",
        linewidth=1.5,
    )

    # plot the the best observed value from each algorithm
    for algo in opt.algos:
        ys = np.vstack(best_vals[algo])
        ax.errorbar(
            iters, ys.mean(axis=0), yerr=ci(ys), label=algo, linewidth=1.5
        )

    ax.set(
        xlabel=f"Number of queries (q = {opt.q}, num_comparisons = {opt.q_comp})",
        ylabel="Best observed value",
        title=f"{opt.dim}-dim weighted vector sum",
    )
    ax.legend(loc="best")

    fig.save(f"pairwise_probit_eubo_q-{opt.q}_NUM-TRIALS-{opt.num_trials}_NUM-BATCHES-{opt.num_batches}_NUM-RESTARTS-{opt.num_restarts}_RAW-SAMPLES-{opt.raw_samples}.png")