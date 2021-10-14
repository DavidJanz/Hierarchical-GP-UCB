"""
Test functions used in experiments.
"""


import argparse
from typing import List
import numpy as np


class PlotParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('bandit', type=str)
        self.add_argument('--grid', type=int, default=30)


class TestFunction:
    def __init__(self, function, bounds: List, fmax: float, fmin: float):
        self._function = function
        self._bounds = bounds
        self._fmax = fmax
        self._fmin = fmin

    def evaluate(self, x):
        x_rescaled = []
        for i, b in enumerate(self._bounds):
            x_rescaled.append(x[i] * (b[1] - b[0]) + b[0])
        f = self._function(*x_rescaled)

        f_zeromax = (f - self._fmax) / (self._fmax - self._fmin)
        return 2 * f_zeromax + 1

    def get_max(self):
        return self._fmax

    def get_ndim(self):
        return len(self._bounds)


test_functions = {}


def _forrester(x1):
    """
    Forrester et al. function
    https://www.sfu.ca/~ssurjano/forretal08.html
    Domain: [0, 1]
    x* = 0.757249
    f(x*) = 6.02074
    min: -15.830

    Source: https://www.wolframalpha.com/input/?i=minimise+%286x-2%29%5E2+*+sin%2812x-4%29+between+0+and+1
    """
    f = (6 * x1 - 2) ** 2 * np.sin(12 * x1 - 4)
    return -f


test_functions['forrester'] = TestFunction(_forrester, bounds=[[0, 1]],
                                           fmax=6.02074, fmin=-15.83)


def _eggholder(x1, x2):
    """
    Eggholder function
    https://www.sfu.ca/~ssurjano/egg.html
    Domain: [-512, 512]^2
    x* = (512, 404.2319)
    f(x*) = 959.6407
    min: -507.874

    Source: https://www.wolframalpha.com/input/?i=maximise+-%28b%2B47%29*sin%28sqrt%28abs%28b%2Ba%2F2%2B47%29%29%29+-+a*sin%28sqrt%28abs%28a-%28b%2B47%29%29%29%29+for+-512+%3C%3D+a+%3C%3D+512+and+-512+%3C%3D+b+%3C%3D+512
    """
    return -(-(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + x2 + 47))) - x1 * np.sin(
        np.sqrt(abs(x1 - (x2 + 47)))))


test_functions['eggholder'] = TestFunction(_eggholder,
                                           bounds=[(-512, 512), (-512, 512)],
                                           fmax=959.6407, fmin=-507.874)


def _rosenbrock(x1, x2):
    """
    Rosenbrock function
    https://www.sfu.ca/~ssurjano/rosen.html
    Domain [-2.048, 2.048]^2
    x* = (1, 1)
    f(x*) = 0
    min: -3905.93

    Source: https://www.wolframalpha.com/input/?i=maximise++100%28b+-+a%5E2%29%5E2+%2B+%28a-1%29%5E2+for+-2.048+%3C%3D+a+%3C%3D+2.048+and+-2.048+%3C%3D+b+%3C%3D+2.048
    """
    f = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return -f


test_functions['rosenbrock'] = TestFunction(_rosenbrock,
                                            bounds=[(-2.048, 2.048),
                                                    (-2.048, 2.048)],
                                            fmax=0, fmin=-3905.93)


def _camel(x1, x2):
    """
    Six-hump camel function
    https://www.sfu.ca/~ssurjano/camel6.html
    Domain: [-3, 3] x [-2, 2]
    x* = (0.0898, -0.7126) and (-0.0898, 0.7126)
    f(x*) = 1.0316

    min: -162.9
    Source: https://www.wolframalpha.com/input/?i=maximise++%284+-+2.1a%5E2+%2B+a%5E4%2F3%29*a%5E2+%2B+a*b+%2B+%28-4%2B4*b%5E2%29*b%5E2+for+-3+%3C%3D+a+%3C%3D+3+and+-2+%3C%3D+b+%3C%3D+2
    """
    f = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 \
        + (-4 + 4 * x2 ** 2) * x2 ** 2
    return -f


test_functions['camel'] = TestFunction(_camel, bounds=[(-3, 3), (-2, 2)],
                                       fmax=1.0316, fmin=-162.9)


def _schaffer(x1, x2):
    """SCHAFFER FUNCTION N.2
    https://www.sfu.ca/~ssurjano/schaffer2.html
    Domain: [-2.5, 2.5]^2
    x* = (0, 0)
    f(x*) = 0

    min: -0.9984
    Source: https://www.wolframalpha.com/input/?i=maximise+0.5+%2B+%28%28sin%28a%5E2-b%5E2%29%29%5E2+-+0.5%29+%2F+%281+%2B+0.001*%28a%5E2%2Bb%5E2%29%29%5E2+for+-100+%3C%3D+a+%3C%3D+100+and+-100+%3C%3D+b+%3C%3D+100
    """
    top = np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5
    bottom = (1 + 0.001 * (x1 ** 2 + x2 ** 2))
    return - (0.5 + top / bottom)


test_functions['schaffer'] = TestFunction(_schaffer,
                                          bounds=[(-2.5, 2.5), (-2.5, 2.5)],
                                          fmax=0, fmin=-0.9984)


def _bukin(x1, x2):
    """Bukin N.6
    https://www.sfu.ca/~ssurjano/bukin6.html
    Domain: [-15, -5] x [-3, 3]
    x* = (-10, 1)
    f(x*) = 0

    -229.18
    Source: https://www.wolframalpha.com/input/?i=maximise+100*sqrt%28abs%28b-0.01*a%5E2%29%29+%2B+0.01*abs%28a%2B10%29+for+-15+%3C%3D+a+%3C%3D+-5+and+-3+%3C%3D+b+%3C%3D+3
    """
    return - 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)


test_functions['bukin'] = TestFunction(_bukin, bounds=[(-15, 5), (-3, 3)],
                                       fmax=0, fmin=-229.18)
