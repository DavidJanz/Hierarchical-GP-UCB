"""
Algorithm implementations. GP-UCB is implemented as Hierarchical GP-UCB with
a single cover element.

PGPUCB-ST should be taken as the default Hierarchical GP-UCB algorithm.
See chapter 5 of thesis for details.

Careful: dependence in SupKernelUCB on RKHS norm may be incorrect.
Experiments in thesis used RKHS norm = 1, so this was not an issue.
Need to recheck dependence with paper.
"""

import numpy as np
import gp_regression
import pgp_ucb


class BanditAlg:
    """Interface for bandit algorithm."""
    def __init__(self, horizon: int, grid: np.array):
        self._horizon = horizon
        self._grid = grid

    def select_arm(self):
        raise NotImplementedError

    def update(self, arm, value):
        raise NotImplementedError


class UniformAlg(BanditAlg):
    """Samples arms uniformly."""
    def select_arm(self):
        index = np.random.randint(len(self._grid))
        return self._grid[index]

    def update(self, arm, value):
        pass


class UCB(BanditAlg):
    """Improved GP-UCB algorithm."""
    def __init__(self, horizon: int, grid: np.array,
                 kernel: gp_regression.MaternKernel,
                 index_fn: gp_regression.UCBIndex):
        super().__init__(horizon, grid)

        self._index_fn = index_fn
        self._kernel = kernel

        _, ndim = grid.shape
        self._intervals = [pgp_ucb.Interval(pgp_ucb.Bound([[0, 1]] * ndim),
                                            kernel, index_fn, grid)]

    def select_arm(self):
        _, argmax = self._intervals[0].get_ucb()
        return argmax

    def update(self, arm, value):
        self._intervals[0].update(arm, value)


class PGPUCB(UCB):
    """Raw Partitioned GP-UCB algorithm."""
    def __init__(self, horizon: int, grid: np.array,
                 kernel: gp_regression.MaternKernel,
                 index_fn: gp_regression.UCBIndex):
        super().__init__(horizon, grid, kernel, index_fn)

        self._successive_tightening = False

    def update(self, arm, value):
        super().update(arm, value)

        if self._intervals[0].needs_split:
            interval = self._intervals.pop(0)
            self._intervals += interval.split(
                successive_tightening=self._successive_tightening)

        self._intervals.sort(key=lambda interval: interval.get_ucb()[0],
                             reverse=True)


class PGPUCB_ST(PGPUCB):
    """Partitioned GP-UCB with successive tightening."""
    def __init__(self, horizon: int, grid: np.array,
                 kernel: gp_regression.MaternKernel,
                 index_fn: gp_regression.UCBIndex):
        super().__init__(horizon, grid, kernel, index_fn)

        self._successive_tightening = True


class SupKernelUCB(BanditAlg):
    """SupKernelUCB algorithm. See note at top of file."""
    def __init__(self, horizon, grid, kernel, delta):
        super().__init__(horizon, grid)

        self._kernel = kernel
        self.gprs = []
        self.beta = (2 * np.log(2 * horizon * len(grid) / delta)) ** (1/2)

        self.arms = self._grid.copy()
        self.threshold = horizon ** (-1/2)
        self.s = 1
        self.store = False
        self.step = 0

    def select_arm(self):
        self.step += 1
        while True:
            arms = self.arms

            preds = self._get_gpr(self.s).predict(arms)
            b = self.beta * preds.sigma

            width_threshold = 2 ** (-self.s)

            select_cond = (b > width_threshold)
            if select_cond.sum() > 0:
                index = np.random.choice(np.flatnonzero(select_cond))
                bval = b[index]
                self.store = True
                return arms[index]

            ucbs = preds.mu + b
            if (b < self.threshold).all():
                return ucbs.argmax()

            ucb_threshold = ucbs.max() - 2 * 2 ** (-self.s)

            self.arms = arms.copy()[ucbs >= ucb_threshold]
            self.s += 1

    def update(self, arm, value):
        if self.store:
            self._get_gpr(self.s).update(arm, value)
            self.store = False

    def _get_gpr(self, s):
        if len(self.gprs) <= s:
            self.gprs += [gp_regression.OnlineGPR(self._kernel)]
        return self.gprs[s - 1]
