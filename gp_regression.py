"""
Code for conjugate Gaussian process regression using online Cholesky updates.
"""

from collections import namedtuple

import gpytorch
import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

# prediction tuple has the mean prediction, confidence width and infogain
prediction = namedtuple('prediction', ('mu', 'sigma', 'gamma'))


def compute_rkhs_norm(function_values, grid, kernel):
    """Computes RKHS norm of discretised function."""
    f = torch.from_numpy(np.array(function_values)[:, None])
    K = kernel(torch.from_numpy(grid))
    L = torch.linalg.cholesky(K + torch.eye(K.shape[0]) * 1e-14)
    trig, _ = f.triangular_solve(L, upper=False)
    rkhs_norm = (trig ** 2).sum().sqrt().item()
    return rkhs_norm


def get_cholesky(kernel, X):
    """Computes Cholesky of Gram matrix kernel(X, X)."""
    X = torch.from_numpy(np.array(X))

    with torch.no_grad():
        return torch.linalg.cholesky(
            kernel(X) + torch.eye(len(X)) * kernel.alpha)


def get_prior_cholesky(kernel):
    """Returns default Cholesky for the prior."""
    with torch.no_grad():
        return (torch.ones(1, 1) * (1 + kernel.alpha)).sqrt()


def update_cholesky(kernel, X, Xnew, L):
    """Online Cholesky update."""
    X = torch.from_numpy(np.array(X))
    Xnew = torch.from_numpy(np.array(Xnew))

    with torch.no_grad():
        Xnew = torch.from_numpy(np.array(Xnew))

        n = L.shape[0]
        a = kernel(X, Xnew)
        c, _ = a.triangular_solve(L, upper=False)
        d = (1 + kernel.alpha - c.t() @ c).sqrt()
        Lnew = torch.zeros(n + 1, n + 1)
        Lnew[-1, :-1] = c.t()
        Lnew[-1, -1] = d
        Lnew[:-1, :-1] = L
        return Lnew


def effective_dimension(L, K):
    """Compute effective dimension from Cholesky and Gram matrix."""
    return (K @ L.cholesky_inverse()).trace().item()


def information_gain(L, kernel):
    """Compute information gain from Cholesky and kernel function"""
    return 0.5 * (L ** 2).diag().log().sum().item() - len(L) / 2 * np.log(
        kernel.alpha)


def gp_predict(X, Y, Xstar, kernel, L):
    """Predict at locations Xstar."""
    Y = torch.from_numpy(np.array(Y))
    X = torch.from_numpy(np.array(X))
    Xstar = torch.from_numpy(np.array(Xstar))

    with torch.no_grad():
        diag = kernel(Xstar, Xstar, diag=True)

        if not len(X):
            return prediction(torch.zeros(Xstar.shape[0]),
                              diag, torch.zeros(Xstar.shape[0]))
        Kx = kernel(X, Xstar)
        A, _ = Kx.triangular_solve(L, upper=False)
        V, _ = Y.triangular_solve(L, upper=False)
        fmean = A.t() @ V
        fvar = diag - (A ** 2).sum(0)
        info_gain = torch.ones(Xstar.shape[0]) * information_gain(L, kernel)
        return prediction(fmean.squeeze(-1), fvar.sqrt(), info_gain)


class OnlineGPR:
    """Wrapper for online Cholesky updates and prediction."""
    def __init__(self, kernel):
        self._kernel = kernel
        self._X = []
        self._Y = []
        self._L = None

    def update(self, x_new, y_new):
        if self._L is None:
            self._L = get_cholesky(self._kernel, [x_new])
        else:
            self._L = update_cholesky(self._kernel, self._X, [x_new], self._L)

        self._X.append(x_new)
        self._Y.append([y_new])

    def predict(self, locations):
        L = get_prior_cholesky(self._kernel) if self._L is None else self._L
        predictions = gp_predict(self._X, self._Y, locations, self._kernel, L)
        return predictions


class UCBIndex:
    """The \beta_t multiplier for GP-UCB."""
    def __init__(self, delta, noise_norm, rkhs_norm, factor=1.0):
        self._delta = delta
        self._noise_norm = noise_norm
        self._rkhs_norm = rkhs_norm
        self._factor = factor

    def __call__(self, mu, sigma, gamma):
        beta = self._rkhs_norm + self._noise_norm * \
               np.sqrt(2 * (gamma + 1 + np.log(self._factor / self._delta)))
        return mu + beta * sigma


class MaternKernel:
    """Wrapper for Matern kernel that bypasses gpytorch lazy evaluation."""
    def __init__(self, nu=2.5, alpha=1.0, ell=None, rand_ell=False):
        self.k = gpytorch.kernels.MaternKernel(nu)
        if rand_ell:
            self.k.lengthscale = np.random.beta(2, 5)
        elif ell:
            self.k.lengthscale = ell
        else:
            self.k.lengthscale = 1
        self.nu = nu
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        if 'diag' in kwargs and kwargs['diag']:
            return self.k(*args, **kwargs)
        return gpytorch.lazy.LazyTensor.evaluate(self.k(*args, **kwargs))
