"""
Interface to bandit problems.
"""

import numpy as np
import test_fns


def gaussian_noise_fn(noise):
    return lambda: np.random.randn() * noise


def uniform_noise_fn(noise):
    return lambda: (np.random.rand(1).item() * 2 - 1) * noise


_noises = {
    'gaussian': gaussian_noise_fn,
    'uniform': uniform_noise_fn,
}


class Bandit:
    def __init__(self, test_fn_name: str, noise_norm: float, noise_type: str):
        self._test_fn = test_fns.test_functions[test_fn_name]

        self._noise_fn = _noises[noise_type](noise_norm)
        self._losses = []

    def pull(self, arm):
        f = self._test_fn.evaluate(arm)
        self._losses.append(1-f)

        return f + self._noise_fn()

    def get_losses(self):
        return self._losses

    def get_regrets(self):
        return np.cumsum(self._losses)

    def get_ndim(self):
        return self._test_fn.get_ndim()

    def reset(self):
        self._losses = []

    def evaluate(self, arm):
        return self._test_fn.evaluate(arm)
