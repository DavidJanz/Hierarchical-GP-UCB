"""
Runs all experiments included in thesis using multiprocessing.
Data is saved in ./data folder in pandas format.
"""

import os

# Force everything to run single threaded.
# Needs to be done before certain imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import argparse
import itertools
import torch
import copy
import multiprocessing
import numpy as np

import bandits
import gp_regression
import util
import algorithms


class MainArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('name', type=str)
        self.add_argument('--algorithm', type=str, default='uniform')
        self.add_argument('--nu', type=float, default=0.5)
        self.add_argument('--ell', type=float, default=1.0)
        self.add_argument('--delta', type=float, default=0.05)
        self.add_argument('--noise_norm', type=float, default=1.0)
        self.add_argument('--horizon', type=int, default=1000)
        self.add_argument('--problem', type=str, default='')
        self.add_argument('--points_per_dim', type=int, default=100)
        self.add_argument('--noise_type', type=str, default='gaussian')
        self.add_argument('--seed', type=int, default=0)


def args_to_name(args):
    """Creates file name by concatenating arguments."""
    name = ",".join([f"{k}={v}" for k, v in vars(args).items()])
    return name


def opt_dict_to_configs(args, opt_dict):
    """Creates args for multirun from default args and dictionary of options."""
    opt_lists = list(itertools.product(*[
        [(k, vs) for vs in v] for k, v in opt_dict.items()]))

    args_dict = vars(args)
    args_list = []
    for opt_list in opt_lists:
        new_args_dict = copy.copy(args_dict)
        for k, v in opt_list:
            new_args_dict[k] = v
        args_list.append(argparse.Namespace(**new_args_dict))
    return args_list


def run_experiment(args):
    """Runs experiment with args."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    grid = util.make_grid([[0, 1]] * 2, args.points_per_dim)
    _, ndim = grid.shape

    bandit = bandits.Bandit(args.problem, args.noise_norm, args.noise_type)

    if args.algorithm == 'uniform':
        alg = algorithms.UniformAlg(horizon=args.horizon, grid=grid)

    elif args.algorithm == 'ucb':
        kernel = gp_regression.MaternKernel(nu=args.nu, alpha=1.0, ell=args.ell)
        alg = algorithms.UCB(horizon=args.horizon, grid=grid, kernel=kernel,
            index_fn=gp_regression.UCBIndex(
        delta=args.delta, noise_norm=args.noise_norm, rkhs_norm=1.0,
        factor=1.0))

    elif args.algorithm == 'supkernelucb':
        kernel = gp_regression.MaternKernel(nu=args.nu, alpha=args.noise_norm,
                                            ell=args.ell)
        alg = algorithms.SupKernelUCB(args.horizon, grid, kernel,
                                      delta=args.delta)

    elif args.algorithm == 'pgpucb':
        kernel = gp_regression.MaternKernel(nu=args.nu, alpha=1.0, ell=args.ell)
        factor = 4 * (args.horizon + 1) ** (ndim / (2 * kernel.nu))
        alg = algorithms.PGPUCB(horizon=args.horizon, grid=grid, kernel=kernel,
               index_fn=gp_regression.UCBIndex(
        delta=args.delta, noise_norm=args.noise_norm, rkhs_norm=1.0,
        factor=factor))

    elif args.algorithm == 'pgpucb_st':
        kernel = gp_regression.MaternKernel(nu=args.nu, alpha=1.0, ell=args.ell)
        factor = 4 * (args.horizon + 1) ** (ndim / (2 * kernel.nu))
        alg = algorithms.PGPUCB_ST(horizon=args.horizon, grid=grid,
                                  kernel=kernel, index_fn=gp_regression.UCBIndex(
        delta=args.delta, noise_norm=args.noise_norm, rkhs_norm=1.0,
        factor=factor))
    else:
        raise ValueError

    try:
        regrets, times = util.run_alg(args.horizon, bandit, alg)
    except KeyboardInterrupt:
        exit()

    return vars(args), {'regret': regrets, 'time': times}


if __name__ == '__main__':
    num_processes = max(multiprocessing.cpu_count() - 1, 1)
    multiprocessing.set_start_method('spawn')
    args = MainArgParser().parse_args()

    # will run each combination of list elements
    config = {
        'problem': ['eggholder', 'rosenbrock', 'camel', 'schaffer', 'bukin'],
        'algorithm': ['supkernelucb', 'ucb', 'pgpucb', 'pgpucb_st'],
        'noise_norm': [0.05, 0.5],
        'seed': list(range(10)),
        'ell': [1.0],
        'nu': [1/2],
    }

    args_list = opt_dict_to_configs(args, config)
    all_results = []
    i = 0
    try:
        with multiprocessing.Pool(num_processes, maxtasksperchild=1) as executor:
            for (args_dict, results) in executor.imap_unordered(run_experiment,
                                                      args_list):
                i += 1
                all_results += [
                    {'step': i + 1, 'regret': r, 'time': t, **args_dict}
                    for i, (r, t) in
                    enumerate(zip(results['regret'], results['time']))]

                print(f"{i}/{len(args_list)} -- "
                      f"{args_dict['problem']}: {args_dict['algorithm']},"
                      f" regret {results['regret'][-1]:.1f}, time "
                      f"{results['time'][-1]:.1f}s")
    except KeyboardInterrupt:
        # Provides clean exit on keyboard interrupt.
        print("Caught keyboard interrupt, exiting.")
        exit()

    # data goes in ./data in pandas format
    results_df = pd.DataFrame(all_results)
    os.makedirs('data', exist_ok=True)
    results_df.to_pickle(f'data/{args.name}.pkl')
