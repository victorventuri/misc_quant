"""
Scripts to simulate Itô process.

The Itô process is effectively a generalization of the Wiener process, in which the drift and diffusion can depend on
both the present time and on the present value of the process.

Similarly, it resembles the Gaussian process we already define, but it scales the variance of the stochastic steps by dt
"""
from typing import Union

import numpy as np

from .gaussian_random_walk import GaussianRandomWalk


class ItoProcess(GaussianRandomWalk):
    """
    Base definition of an Itô process
    """
    def get_drift(self, time: float, x: float) -> Union[float, np.ndarray]:
        return self._drift(time=time, x=x)

    def _get_step_cov(self, time: float, x: float) -> Union[float, np.ndarray]:
        dt = self.dt
        return dt * np.atleast_2d(self._stoc_step_cov(time=time, x=x))
