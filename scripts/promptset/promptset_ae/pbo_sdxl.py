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
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions.errors import OptimizationGradientError
from botorch.exceptions.errors import ModelFittingError
from botorch.optim.initializers import gen_batch_initial_conditions
from models.pairwise_gp_new import PairwiseGP
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt

from models.multiwise_gp import MultiwiseGP, MultiwiseLaplaceMarginalLogLikelihood

from utils.obj_utils import get_objective
from utils.acf_utils import get_acf
from utils.obj_utils import *
from utils.gp_utils import generate_batch_initial_conditions, init_and_fit_model,eval_kt_cor
from utils.general_utils import set_all_seeds, plot_results, make_plot_path, save_results, save_config
from configs import MultiBOConfig

# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")

def create_population_models(utility, gen_comp, gen_data, likelihood, opt):

    mll_tot = []
    model_tot = []
    for i in range(opt.num_population_models):
        train_X, train_y = gen_data(utility, opt.num_population_data, dim=opt.dim, device=opt.device,opt=opt)
        train_comp, train_ch = gen_comp(train_y, opt.m_pop, T=opt.T, noise=opt.noise, device=opt.device,title=opt.title)
        mll, model = init_and_fit_model(train_X, train_comp, likelihood, ch=train_ch, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  
        mll_tot.append(mll)
        model_tot.append(model)
    
    return mll_tot, model_tot
def create_folder_path(opt):
    if opt.multi:
            pp = os.path.join(opt.output_path,f"Multiwise")
            assert opt.q == opt.T 
    else:
        pp = os.path.join(opt.output_path,f"Pairwise")
        assert opt.T == 2
    
    if opt.obj_name not in opt.image_models:
        pp = os.path.join(pp,f"dim-{opt.dim}")

    if opt.logit:
        opt.type_likelihood=True
        pp = os.path.join(pp,f"Logit-{opt.acf}")
    else:
        pp = os.path.join(pp,f"Probit-{opt.acf}")
        if not opt.multi:
            opt.type_likelihood=False
        else:
            raise ValueError(f"No probit model for MultiwiseGP")
    
    if opt.obj_name in opt.image_models:
        pp = os.path.join(pp,f"{opt.prompts}",f"seed-{opt.img_seed}")
    return pp

@pyrallis.wrap()
def start(opt: MultiBOConfig):
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.seed = 42
    set_all_seeds(opt.seed)
    # torch.manual_seed(123)
    # initialize modules
    utility, optimum, obj = get_objective(opt.obj_name,opt.mode)
    if opt.obj_name not in opt.image_models:
        gen_comp = generate_comparisons_multi if opt.multi else generate_comparisons_pair
        gen_data = generate_data
        make_new = make_new_data
        kwargs_gen_data = None
        kwargs_gen_comp = None
        kwargs_make_new = None
    else:
        gen_comp = generate_comparisons_image
        gen_data = generate_data_image
        make_new = make_new_data_image
        kwargs_gen_data = {'opt':opt, 'type':'sobol' if opt.init_sobol else 'uniform'}
        kwargs_gen_comp = {'obj':obj}
        kwargs_make_new = {'opt':opt,'obj':obj}
    
    likelihood = MultiwiseLaplaceMarginalLogLikelihood if opt.multi else PairwiseLaplaceMarginalLogLikelihood
    
    
    acfs = opt.acf.split("+")
    # acfs.append("rand")
    opt.acf_algos = acfs
    acf_func_lst = get_acf(opt.acf_algos, opt)

    opt.output_path = create_folder_path(opt)
    opt.cross_maps_path = os.path.join(opt.output_path,"masks")
    ss = []
    for key, values in opt.attn_blk.items():
        if values: 
            combined_values = "-".join(values)
            ss.append(f"{key}_{combined_values}")
    ss = "--".join(ss)
    comp = opt.feats
    opt.output_path = os.path.join(opt.output_path,"acf-fixed-new",opt.edit_type,"single-mask",f"{opt.feat_blocks}-blk","test",f"res-{opt.res_blk}--{ss}",f"T-edit-{opt.t_edit}_Delta-{opt.delta}-{comp}")
    os.makedirs(opt.output_path,exist_ok=True)
    
    if opt.prev_winner:
        opt.q = 1
    
    ## generate random datapoints and verify model fitting
    # train_X, train_y = gen_data(utility, opt.num_initial_samples, dim=opt.dim, device=opt.device,opt=opt,kwargs=kwargs_gen_data) #
    # train_comp, train_ch = gen_comp(train_y, opt.m, T=opt.T, noise=opt.noise, device=opt.devicekwargs=kwargs_gen_comp)
    # mll, model = init_and_fit_model(train_X, train_comp, likelihood, ch=train_ch, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  
    
    # initial evals
    best_vals = {}  # best observed values
    for algo in opt.acf_algos:
        best_vals[algo] = []

    # average over multiple trials
    for i in range(opt.num_trials):
        set_all_seeds(opt.random_seeds[i])
        seed = opt.random_seeds[i]
        opt.seed = seed
        print(" ")
        print(f"Trial with seed = {seed}")
        print(" ")       
        
        data = {}
        models = {}
        
        title = f"PROMPT= {opt.prompts} T-EDIT= {opt.t_edit} \
        DELTA= {opt.delta} IMG_SEED= {opt.img_seed} EDIT_TYPE= {opt.edit_type} EDIT_BLOCKS= res-{opt.res_blk}-{comp}-{ss} BEFORE_SOFTMAX= {opt.before_softmax} BO_SEED= {opt.seed} \
        INIT_SAMPLES= {opt.num_initial_samples} INIT_COMP= {opt.m} ACF= {acfs[0]} NUM_TRIALS= {opt.num_trials} \
        NUM_BATCHES= {opt.num_batches} Q= {opt.q} PREV_WINNER= {opt.prev_winner}"
        
        opt.title = title
        for algo in opt.acf_algos:
            save_config(algo,i,seed,opt)
        # Generate initial datapoints and comparisons
        if not opt.load_prefit_model:
            init_X, init_y = gen_data(utility, opt.num_initial_samples, dim=opt.dim, device=opt.device,opt=opt,kwargs=kwargs_gen_data)
            # print(init_X.shape,init_y.shape)
            kwargs_gen_comp['title'] = opt.title
            comparisons, choices = gen_comp(init_y, opt.m, T=opt.T, noise=opt.noise, device=opt.device, kwargs=kwargs_gen_comp) #

        # X are within the unit cube
        bounds = torch.stack([opt.lim[0]*torch.ones(opt.dim), opt.lim[1]*torch.ones(opt.dim)]).to(opt.device)

        # acf_init_X_new, acf_init_y_new = gen_data(utility, opt.num_restarts * opt.q, dim=opt.dim, device=opt.device,opt=opt,kwargs=kwargs_gen_data)
        acf_init_X_new=None
        batch_initial_conditions = generate_batch_initial_conditions(
        bounds=bounds,
        num_restarts=opt.num_restarts,
        q=opt.q,
        existing_points=acf_init_X_new,  # optional
        device=opt.device,)
        

        # for_rand_algo_X, for_rand_algo_y = gen_data(utility, opt.q*opt.num_batches, dim=opt.dim, device=opt.device,opt=opt,kwargs=kwargs_gen_data)

        # we make additional num_batches comparison queries after the initial observation
        for j in range(1, opt.num_batches + 1):
            for algo, acf_f in zip(opt.acf_algos,acf_func_lst):
                if j == 1:
                    # fit model on initial data and store best observation
                    best_vals[algo].append([])
                    if opt.load_prefit_model:
                        dat = torch.load(opt.prefit_model_data_path)
                        data[algo] = dat["data"]
                        # model.load_state_dict(kk['models'])
                    else:
                        data[algo] = (init_X, init_y, comparisons, choices)
                    _, models[algo] = init_and_fit_model(init_X, comparisons, likelihood, ch=choices, device=opt.device,type_likelihood=opt.type_likelihood, multi=opt.multi)  

                    if opt.obj_name not in opt.image_models:
                        best_next_y = utility(init_X, device=opt.device).max().item()
                        best_vals[algo][-1].append(best_next_y)
                    else:
                        # best_next_y = utility(init_X, opt)
                        best_X, best_image = find_best_image(init_X, init_y,title=opt.title,kwargs={"obj":opt.obj})
                        best_vals[algo][-1].append(best_image)
                        prev_winner = best_X if opt.prev_winner else None
                        
                if "TAF" in algo:
                    opt.increment = True
                    _, pop_models = create_population_models(utility,gen_comp,gen_data,likelihood, opt)
                    # 2s-TAF-qEI
                    if "2s" in algo:
                        if opt.ref_points_path is None:
                            opt.ref_points = init_X[:opt.num_ref_points].to(opt.device)
                        else:
                            opt.ref_points = torch.load(opt.ref_points_path,device=opt.device)
                        acq_func = acf_f(
                                            models[algo],
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
                        acq_func = acf_f(models[algo],
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
                        model=models[algo],
                        inner_acq_cls=qExpectedImprovement,
                        num_fantasies=opt.num_fantasies,
                        ref_points=opt.ref_points,  # optional
                        device=opt.device,
                        T=opt.T,
                        multi=opt.multi
                    )
                    # acq_func = TwoStepLookahead(models[algo], num_fantasies=opt.num_fantasies, use_taf=False, ref_points=opt.ref_points)
                
                elif "EUBO" in algo:
                    opt.increment = False
                    acq_func = acf_f(pref_model=models[algo],previous_winner=prev_winner)
                elif "UCB" in algo:
                    opt.increment = False
                    weights = torch.tensor([1.0], device=init_X.device, dtype=init_X.dtype)
                    pt = ScalarizedPosteriorTransform(weights=weights)
                    acq_func = acf_f(model=models[algo],beta=opt.beta,posterior_transform=pt)
                elif "qEI" in algo:
                    opt.increment = False
                    # acq_func = acf_f(model=models[algo], best_f = init_y.max()) 
                    acq_func = acf_f(model=models[algo], X_baseline = data[algo][0])       

                # optimize the fitted model to find next sample
                model = models[algo]
                if algo != "rand" and algo != "2s-qEI":
                
                    batch_init = gen_batch_initial_conditions(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=opt.q,
                        num_restarts=opt.num_restarts,
                        raw_samples=opt.raw_samples,
                    )
                    try:
                        # optimize and get new observation
                        next_X, acq_val = optimize_acqf(
                            acq_function=acq_func,
                            bounds=bounds,
                            q=opt.q,
                            num_restarts=opt.num_restarts,
                            raw_samples=opt.raw_samples,
                            # batch_initial_conditions=batch_initial_conditions,
                            # options={"batch_limit": 40},
                        )
                        if opt.increment:
                            acq_func.increment_iteration()
                            # acq_func.source_models = [model,pop_models[0]]
                    except:
                        print("NaN gradient, retrying with fresh init...")
                        batch_initial_conditions = None  # let BoTorch regenerate safely
                        next_X, acq_val = optimize_acqf(
                            acq_function=acq_func,
                            bounds=bounds,
                            q=opt.q,
                            num_restarts=opt.num_restarts,
                            raw_samples=opt.raw_samples,
                        )

                # else:
                #     # randomly sample data
                #     next_X = for_rand_algo_X[(j-1)*opt.q:(j)*opt.q]
                  
                # update data
                # refit models                
                X, y, comps, chs = data[algo]
                kwargs_make_new['title'] = opt.title
                if opt.prev_winner:
                    next_X = torch.vstack([next_X,best_X])
                X, y, comps, chs = make_new(utility, X, next_X, y, comps, opt.q_comp, choices=chs, multi=opt.multi, T=opt.T, device=opt.device, noise=opt.noise, kwargs=kwargs_make_new)
                data[algo] = (X, y, comps, chs)
                try:
                    _, models[algo] = init_and_fit_model(X, comps, likelihood, ch=chs, device=opt.device, type_likelihood=opt.type_likelihood, multi=opt.multi)
                except ModelFittingError:
                    print(f"Fit attempt failed, retrying...")
                    X = X + 1e-6*torch.randn_like(X)
                if hasattr(acq_func ,"model"):
                    acq_func.model = models[algo]
                # record the best observed values so far
                if opt.obj_name not in opt.image_models:
                    max_val = utility(X, device=opt.device).max().item()
                    best_vals[algo][-1].append(max_val)
                    print(max_val)
                else:
                    # max_y = utility(X, opt)
                    best_X, max_image = find_best_image(X,y,title=opt.title,kwargs={"obj":opt.obj})
                    best_vals[algo][-1].append(max_image)
                    prev_winner = best_X if opt.prev_winner else None
                
                print(" ")
                print(f"Completed -------- Algo={algo},Trials={i},batches={j}")
                print(" ")
                
                
        if opt.save_results:
            print(" ")
            print("Saving Results")
            print(" ")
            if i % opt.save_every_trial == 0:
                if opt.ref_points is not None:
                    opt.ref_points = opt.ref_points.tolist()
                save_results(data,models,best_vals, i, seed, opt)
        


    if opt.plotting:
        opt.plot_path = make_plot_path(opt)
        plot_results(utility,optimum(opt.dim,opt.device), best_vals, opt)


if __name__ == "__main__":
    start()
    
    