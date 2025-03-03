"""
Scripts to simulate a Wiener process, characterized by
- W(0) = 0 almost surely
- W has independent increments, that is, W(t+u) - W(t) (u>=0) are independent of W(s) for all s<t
- W has Gaussian increments: W(t+u) - W(t) ~ N(0,u)
- W is continuous in t

The continuity in time is the hardest part to get computationally. Wikipedia's suggestion is to create a collection of
i.i.d epsilons from N(0,1) and define, for t in [0,1],
W_n(t) = (1/sqrt(n)) * Sum_{i=1}^{\floor{n*t}} epsilon_i

In this case, as n->infinity, this approaches a true Wiener process.

We will instead opt for a simpler approach: we specify a number of evenly spaced steps, and the interval dt between the
values we wish to generate. We then create i.i.d. epsilons from N(0,dt), and cumulative sum them. Note that this
immediately guarantees the first two conditions above, and also the third one:
W(t+u) = W(t+(n*dt)) = W(t) + Sum_{n} epsilon_i
Recall, however, that the variance of the sum of two independent random variables is the sum of their individual
variances. Therefore, the variance of the sum of n epsilons is simple n*dt = u.
"""
from typing import Callable, Union

import numpy as np

from quant_learn.utils.funcs import ConstantFunction
from .gaussian_random_walk import GaussianRandomWalk


class GeneralizedWiener(GaussianRandomWalk):
    """
    Base definition of a generalized Wiener process

    Args:
        infinitesimal_time_interval: infinitesimal time interval to be used in simulations
        drift: drift function
        stochastic_stepper: function that computes the covariance for the stochastic step (but not the dt term!!)
        seed: value used to seed the random number generator
    """
    def __init__(self,
                 infinitesimal_time_interval: float,
                 drift: Callable = ConstantFunction(constant=0.),
                 stochastic_stepper: Callable = ConstantFunction(constant=1.),
                 seed: int = 1234) -> None:
        super().__init__(infinitesimal_time_interval=infinitesimal_time_interval,
                         drift=drift,
                         stochastic_stepper=stochastic_stepper,
                         seed=seed)

    def get_drift(self, time: float, x: float) -> Union[float, np.ndarray]:
        return self._drift(time=time)

    def _get_step_cov(self, time: float, x: float) -> Union[float, np.ndarray]:
        dt = self.dt
        return dt * np.atleast_2d(self._stoc_step_cov(time=time))
