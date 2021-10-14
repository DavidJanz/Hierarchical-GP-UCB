"""
Code for cover elements for Partitioned GP-UCB.
"""

import numpy as np
import util
import itertools

import gp_regression


class Bound:
    """Object representing interval in R^d."""
    def __init__(self, bounds):
        self._bounds = bounds

    def get_mins_maxs(self):
        mins, maxs = zip(*self._bounds)
        return mins, maxs

    def split(self):
        choices = []
        for b1, b2 in self:
            midpoint = (b1 + b2) / 2
            choices.append([[b1, midpoint], [midpoint, b2]])
        return [Bound(v) for v in itertools.product(*choices)]

    def __contains__(self, item):
        mins, maxs = self.get_mins_maxs()
        return np.all(item <= maxs) and np.all(item >= mins)

    def __iter__(self):
        return iter(self._bounds)

    def __repr__(self):
        return self._bounds.__repr__()


class Interval:
    """Object representing cover elements for Partitioned GP-UCB."""
    def __init__(self, bound, kernel, index_fn, grid):
        self.bound = bound
        self.Y = []
        self.X = []
        self.kernel = kernel
        self.index_fn = index_fn
        self.grid = grid
        self._online_gpr = gp_regression.OnlineGPR(kernel)

        self._recompute_ucb()

    def update(self, x, y):
        self.Y.append(y)
        self.X.append(x)

        self._online_gpr.update(x, y)
        self._recompute_ucb()

    def get_ucb(self):
        return self._max_ucb, self._argmax_ucb

    def compute_ucb(self, grid):
        predictions = self._online_gpr.predict(grid)
        ucbs = self.index_fn(*predictions)
        argmax_index = util.argmax(ucbs)

        argmax_ucb = grid[argmax_index]
        max_ucb = ucbs[argmax_index].item()
        return argmax_ucb, max_ucb

    def split(self, successive_tightening=False):
        new_intervals = []
        for new_bound in self.bound.split():
            grid_mask = np.array([x in new_bound for x in self.grid])
            new_grid = self.grid[grid_mask]

            if len(new_grid) == 0:
                continue

            new_interval = Interval(new_bound, self.kernel,
                                    self.index_fn, new_grid)

            if successive_tightening:
                new_interval._argmax_ucb, new_interval._max_ucb = \
                    self.compute_ucb(new_grid)

            new_intervals.append(new_interval)

        return new_intervals

    def _recompute_ucb(self):
        if len(self.grid):
            self._argmax_ucb, self._max_ucb = self.compute_ucb(self.grid)
        else:
            self._argmax_ucb, self._max_ucb = None, float('-inf')

    @property
    def diameter(self):
        return max(b[1] - b[0] for b in self.bound)

    @property
    def num_points(self):
        return len(self.Y)

    @property
    def needs_split(self):
        return self.num_points > self.diameter ** (- 2 * self.kernel.nu)
