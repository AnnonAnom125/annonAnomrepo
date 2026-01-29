import os
import warnings


import numpy as np
import torch
import random
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from models.pairwise_gp_new import PairwiseGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt
from acfs.taf import TAFAcquisition
from acfs.twostep import TwoStepLookaheadPairwise
from acfs.twostep_taf import TwoStepLookahead
from models.multiwise_gp import MultiwiseGP, MultiwiseLaplaceMarginalLogLikelihood

from utils.obj_utils import get_objective
from utils.obj_utils import generate_data, generate_comparisons_pair, generate_comparisons_multi, make_new_data
from utils.gp_utils import generate_batch_initial_conditions, init_and_fit_model,eval_kt_cor
from utils.general_utils import set_all_seeds
from configs import MultiBOConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,5,4,3,2,1,0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Suppress potential optimization warnings for cleaner notebook
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def start(opt: MultiBOConfig):


    set_all_seeds(opt.seed)

    utility = get_objective(opt.obj_name,opt.mode)

    train_X, train_y = generate_data(utility, opt.n, dim=opt.dim, device=opt.device)


    if opt.logit:
        opt.type_likelihood=True
    else:
        opt.type_likelihood=False


    opt.algos = opt.algos.append("rand")

    if opt.multi:
        train_comp, train_ch = generate_comparisons_multi(train_y, opt.m, T=opt.T, noise=opt.noise, device=opt.device)
        likelihood = MultiwiseLaplaceMarginalLogLikelihood

    else:
        train_comp = generate_comparisons_pair(train_y, opt.m, noise=opt.noise, device=opt.device)
        train_ch = None
        likelihood = PairwiseLaplaceMarginalLogLikelihood
    mll, model = init_and_fit_model(train_X, train_comp, likelihood, ch=train_ch, device=opt.device,multi=opt.multi)  


    # verify model fitting
    test_X, test_y = generate_data(utility, opt.n_kendall, dim=opt.dim, device=opt.device)
    kt_correlation = eval_kt_cor(model, test_X, test_y, device=opt.device)

    print(f"Test Kendall-Tau rank correlation: {kt_correlation:.4f}")


    # initial evals
    best_vals = {}  # best observed values
    for algo in opt.algos:
        best_vals[algo] = []



    # average over multiple trials
    for i in range(opt.num_trials):
        # torch.manual_seed(i)
        # np.random.seed(i)
        
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
                    data[algo] = (X, comps)
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