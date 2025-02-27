"""
Collection of scripts to define a Gaussian random walk
"""
from typing import Callable, Union

import numpy as np

from . import GeneralizedMarkovRandomWalk


class GaussianRandomWalk(GeneralizedMarkovRandomWalk):
    """
    Base definition for a Gaussian random walk, in which the infinitesimal stochastic step taken is a zero-mean Gaussian
    random variable with covariance scaled by the stochastic stepper function

    Args:
        infinitesimal_time_interval: infinitesimal time interval to be used in simulations
        drift: drift function
        stochastic_stepper: function that computes the covariance for the stochastic step
        seed: value used to seed the random number generator
    """
    def __init__(self,
                 infinitesimal_time_interval: float,
                 drift: Callable,
                 stochastic_stepper: Callable,
                 seed: int = 1234) -> None:
        super().__init__(infinitesimal_time_interval=infinitesimal_time_interval,
                         drift=drift,
                         stochastic_stepper=stochastic_stepper)
        self.seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value
        self._rng = np.random.default_rng(seed=value)

    def _set_drift(self, drift: Callable) -> None:
        self._drift = drift

    def _set_stochastic_stepper(self, stochastic_stepper: Callable) -> None:
        """
        Auxiliary function to set the diffusion
        """
        self._stoc_step_cov = stochastic_stepper

    def _get_step_cov(self, time: float, x: float) -> np.ndarray:
        return np.atleast_2d(self._stoc_step_cov(time=time, x=x))

    def get_drift(self, time: float, x: float) -> Union[float, np.ndarray]:
        """
        Auxiliary function to get drift at given time and given present value of walk
        """
        return self._drift(time=time, x=x)

    def get_stochastic_step(self, time: float, x: float) -> Union[float, np.ndarray]:
        """
        Auxiliary function to get diffusion at given time and given present value of walk
        """
        covariance = self._get_step_cov(time=time, x=x)
        mu = np.zeros_like(covariance)[0].flatten()  # make it 1D

        return self._rng.multivariate_normal(mean=mu, cov=covariance).squeeze()
