import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyrallis
import numpy as np
import torch
import random
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions.errors import OptimizationGradientError
from botorch.exceptions.errors import ModelFittingError
from models.pairwise_gp_new import PairwiseGP

from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt

from models.multiwise_gp import MultiwiseGP, MultiwiseLaplaceMarginalLogLikelihood

from utils.obj_utils import get_objective
from utils.acf_utils import get_acf
from utils.obj_utils import generate_data, generate_comparisons_pair, generate_comparisons_multi, make_new_data
from utils.gp_utils import generate_batch_initial_conditions, init_and_fit_model,eval_kt_cor
from utils.general_utils import set_all_seeds, plot_results, make_plot_path, save_results
from configs import MultiBOConfig


# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")


def create_population_models(utility, gen_comp, likelihood, opt):

    mll_tot = []
    model_tot = []
    for i in range(opt.num_population_models):
        train_X, train_y = generate_data(utility, opt.num_population_data, dim=opt.dim, device=opt.device,opt=opt)
        train_comp, train_ch = gen_comp(train_y, opt.m_pop, T=opt.T, noise=opt.noise, device=opt.device)
        mll, model = init_and_fit_model(train_X, train_comp, likelihood, ch=train_ch, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  
        mll_tot.append(mll)
        model_tot.append(model)
    
    return mll_tot, model_tot

@pyrallis.wrap()
def start(opt: MultiBOConfig):

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(42)
    # torch.manual_seed(123)
    # initialize modules
    utility, optimum, obj = get_objective(opt.obj_name,opt.mode)
    gen_comp = generate_comparisons_multi if opt.multi else generate_comparisons_pair
    likelihood = MultiwiseLaplaceMarginalLogLikelihood if opt.multi else PairwiseLaplaceMarginalLogLikelihood
    acfs = opt.acf.split("+")
    acfs.append("rand")
    opt.acf_algos = acfs
    acf_func_lst = get_acf(opt.acf_algos, opt)

    if opt.multi:
        pp = os.path.join(opt.output_path,f"Multiwise",f"dim-{opt.dim}")
        assert opt.q == opt.T 
    else:
        pp = os.path.join(opt.output_path,f"Pairwise",f"dim-{opt.dim}")

    

    if opt.logit:
        opt.type_likelihood=True
        pp = os.path.join(pp,f"Logit-{opt.acf}")
    else:
        pp = os.path.join(pp,f"Probit-{opt.acf}")
        if not opt.multi:
            opt.type_likelihood=False
        else:
            raise ValueError(f"No probit model for MultiwiseGP")
    
    opt.output_path = pp
    os.makedirs(opt.output_path,exist_ok=True)
    
    # generate random datapoints and verify model fitting
    train_X, train_y = generate_data(utility, opt.num_initial_samples, dim=opt.dim, device=opt.device,opt=opt) #
    train_comp, train_ch = gen_comp(train_y, opt.m, T=opt.T, noise=opt.noise, device=opt.device)
    mll, model = init_and_fit_model(train_X, train_comp, likelihood, ch=train_ch, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  
    test_X, test_y = generate_data(utility, opt.n_kendall, dim=opt.dim, device=opt.device,opt=opt)
    kt_correlation = eval_kt_cor(model, test_X, test_y, device=opt.device)
    print(f"Test Kendall-Tau rank correlation: {kt_correlation:.4f}")
    # initial evals
    best_vals = {}  # best observed values
    for algo in opt.acf_algos:
        best_vals[algo] = []

    

    # average over multiple trials
    for i in range(opt.num_trials):
        set_all_seeds(opt.random_seeds[i])
        # set_all_seeds(i)
        seed = opt.random_seeds[i]
        opt.seed = seed
        print(f"Trial with seed = {seed}")
        # torch.manual_seed(i)
        # np.random.seed(i)        
        
        data = {}
        models = {}

        # Generate initial datapoints and comparisons
        init_X, init_y = generate_data(utility, opt.num_initial_samples, dim=opt.dim, device=opt.device,opt=opt)
        acf_init_X_new, _ = generate_data(utility, opt.num_restarts * opt.q, dim=opt.dim, device=opt.device,opt=opt)
        # print(init_X.shape,init_y.shape)
        comparisons, choices = gen_comp(init_y, opt.m, T=opt.T, noise=opt.noise, device=opt.device) #

        # X are within the unit cube
        bounds = torch.stack([opt.lim[0]*torch.ones(opt.dim), opt.lim[1]*torch.ones(opt.dim)])

        batch_initial_conditions = generate_batch_initial_conditions(
        bounds=bounds,
        num_restarts=opt.num_restarts,
        q=opt.q,
        existing_points=acf_init_X_new,  # optional
        device=opt.device,)

        for_rand_algo, _ = generate_data(utility, opt.q*opt.num_batches, dim=opt.dim, device=opt.device,opt=opt)

        for algo, acf_f in zip(opt.acf_algos,acf_func_lst):

            # fit model on initial data and store best observation
            best_vals[algo].append([])
            data[algo] = (init_X, comparisons, choices)
            _, models[algo] = init_and_fit_model(init_X, comparisons, likelihood, ch=choices, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  

            best_next_y = utility(init_X, device=opt.device).max().item()
            best_vals[algo][-1].append(best_next_y)

            if "TAF" in algo:
                opt.increment = True
                _, pop_models = create_population_models(utility,gen_comp,likelihood, opt)
                # 2s-TAF-qEI
                if "2s" in algo:
                    if opt.ref_points_path is None:
                        opt.ref_points = init_X[:opt.num_ref_points].to(opt.device)
                    else:
                        opt.ref_points = torch.load(opt.ref_points_path,device=opt.device)
                    acq_func = acf_f(
                                        model,
                                        num_fantasies=opt.num_fantasies,
                                        use_taf=True,
                                        source_models=pop_models,
                                        inner_acq_kwargs={"rho": opt.rho, "d1": opt.d1, "d2": opt.d2},
                                        ref_points=opt.ref_points,  # optional
                                        device=opt.device,
                                        T=opt.T,
                                        multi=opt.multi,
                                    )
                # TAFR-qEI
                else:
                    acq_func = acf_f(model,
                                        source_models=pop_models,
                                        rho=opt.rho,
                                        d1=opt.d1,
                                        d2=opt.d2,
                                        device=opt.device,
                                        T=opt.T,
                                        multi=opt.multi,
                                    )
                
            # 2s-qEI
            elif "2s" in algo:
                opt.increment = False
                if opt.ref_points_path is None:
                    opt.ref_points = init_X[:opt.num_ref_points].to(opt.device)
                else:
                    opt.ref_points = torch.load(opt.ref_points_path,device=opt.device)

                acq_func = acf_f(
                    model=model,
                    inner_acq_cls=qExpectedImprovement,
                    num_fantasies=opt.num_fantasies,
                    ref_points=opt.ref_points,  # optional
                    device=opt.device,
                    T=opt.T,
                    multi=opt.multi,
                )
                # acq_func = TwoStepLookahead(model, num_fantasies=opt.num_fantasies, use_taf=False, ref_points=opt.ref_points)
            
            elif "EUBO" in algo:
                opt.increment = False
                acq_func = acf_f(pref_model=model)
            elif "qEI" in algo:
                opt.increment = False
                acq_func = acf_f(model=model, best_f = init_y.max())       

            # we make additional num_batches comparison queries after the initial observation
            for j in range(1, opt.num_batches + 1):
                
                model = models[algo]
                if algo != "rand" and algo != "2s-qEI":
                    try:
                        # optimize and get new observation
                        next_X, acq_val = optimize_acqf(
                            acq_function=acq_func,
                            bounds=bounds,
                            q=opt.q,
                            num_restarts=opt.num_restarts,
                            raw_samples=opt.raw_samples,
                            batch_initial_conditions=batch_initial_conditions,
                        )
                        if opt.increment:
                            acq_func.increment_iteration()
                            # acq_func.source_models = [model,pop_models[0]]
                    except OptimizationGradientError:
                        print("NaN gradient, retrying with fresh init...")
                        batch_initial_conditions = None  # let BoTorch regenerate safely
                        next_X, acq_val = optimize_acqf(
                            acq_function=acq_func,
                            bounds=bounds,
                            q=opt.q,
                            num_restarts=opt.num_restarts,
                            raw_samples=opt.raw_samples,
                        )

                else:
                    # randomly sample data
                    # next_X, _ = generate_data(utility, opt.q, dim=opt.dim, device=opt.device,opt=opt)
                    next_X = for_rand_algo[(j-1)*opt.q:(j)*opt.q]

                print(f"Algo={algo},Trials={i},batches={j}")
                    
                # update data
                # refit models                
                X, comps, chs = data[algo]
                X, comps, chs = make_new_data(utility, X, next_X, comps, opt.q_comp, choices=chs, multi=opt.multi, T=opt.T, device=opt.device, noise=opt.noise)
                data[algo] = (X, comps, chs)
                try:
                    _, models[algo] = init_and_fit_model(X, comps, likelihood, ch=chs, device=opt.device, type_likelihood=opt.type_likelihood, multi=opt.multi)
                except ModelFittingError:
                    print(f"Fit attempt failed, retrying...")
                    X = X + 1e-6*torch.randn_like(X)
                if hasattr(acq_func ,"model"):
                    acq_func.model = models[algo]
                # record the best observed values so far
                max_val = utility(X, device=opt.device).max().item()
                best_vals[algo][-1].append(max_val)
                print(max_val)
                if algo == '2s-TAF-qEI':
                    print(max_val, next_X)
                    
                
                

        if opt.save_results:
            if i % opt.save_every_trial == 0:
                if opt.ref_points is not None:
                    opt.ref_points = opt.ref_points.tolist()
                save_results(data,models,best_vals, i, seed, opt)
        


    if opt.plotting:
        opt.plot_path = make_plot_path(opt)
        plot_results(utility,optimum(opt.dim,opt.device), best_vals, opt)


if __name__ == "__main__":
    start()
    
    