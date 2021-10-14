import torch
import numpy as np
import itertools
import time


def make_grid(bounds, points_per_dim, keep_edges=False):
    """Creates grid for discretisation."""
    if keep_edges:
        grid = [np.linspace(*b, points_per_dim + 2) for b in bounds]
    else:
        grid = [np.linspace(*b, points_per_dim + 2)[1:-1] for b in bounds]
    return np.array(list(itertools.product(*grid)))


def argmax(values):
    """Argmax with random tie-breaking."""
    argmax_indices = torch.arange(len(values))[values == values.max()]
    rand_index = torch.randint(0, len(argmax_indices), (1,))
    return argmax_indices[rand_index].item()


def run_alg(horizon, bandit, algorithm):
    """Loop for running a bandit algorithm on a problem."""
    bandit.reset()
    total_times = []

    t0 = time.perf_counter()
    for step in range(1, horizon + 1):
        arm = algorithm.select_arm()
        reward = bandit.pull(arm)
        algorithm.update(arm, reward)
        total_times.append(time.perf_counter() - t0)

    regrets = bandit.get_regrets()
    return np.array(regrets), np.array(total_times)



