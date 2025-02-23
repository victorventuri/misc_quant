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
from typing import Optional, Union, List, Tuple

import numpy as np


class WienerProcess1D():
    """
    Base definition of a Wiener process

    Args:
        num_steps: number of steps to take
        dt: interval between steps; defaults to 1.0
        start_time: start time of the process in the same units as dt; defaults to 0.
        seed: seed to be used for the random number generator; defaults to None
    """
    def __init__(self,
                 seed: Optional[int] = None) -> None:
        self.seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value
        self._rng = np.random.default_rng(seed=value)

    @property
    def num_steps(self) -> int:
        return len(self.epsilons)

    def compute_epsilons(self, num_steps: int) -> np.ndarray:
        """
        Auxiliary function to compute epsilons.

        Args:
            num_steps: number of steps (epsilon values) to return
        """
        return np.sqrt(self.dt) * self._rng.normal(size=num_steps)

    def simulate(self,
                 num_steps: int = 100,
                 dt: float = 1.0,
                 start_time: float = 0.,) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to simulate Wiener process

        Args:
            num_steps: number of steps to take; defaults to 100
            dt: interval between steps; defaults to 1.0
            start_time: start time of the process in the same units as dt; defaults to 0.

        Returns:
            tuple corresponding to (time stamps, Wiener values)
        """
        self.dt = dt
        self.t0 = start_time
        self.time_stamps = np.linspace(start_time, start_time + (num_steps * dt), num_steps+1)
        self.epsilons = self.compute_epsilons(num_steps=num_steps)
        self.base_values = np.hstack((0., np.cumsum(self.epsilons)))
        return self.get_time_series()

    def get_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns time series of underlying Wiener process

        Returns:
            tuple corresponding to (time stamps, Wiener values)
        """
        return self.time_stamps, self.base_values

    def __call__(self, t: Union[float, List[float], np.ndarray]) -> Union[float, List[float], np.ndarray]:
        if getattr(self, 't0', None) is None:
            num_steps = 100
            dt = t / num_steps
            _, _ = self.simulate(num_steps=num_steps,
                                 dt=dt)
        if t < self.t0:
            raise ValueError(f'Querying time {t} prior to start time of {self.t0}')

        if t > self.time_stamps[-1]:
            # Create a new Wiener process that reaches the target time
            num_steps = int(np.ceil((t - self.time_stamps[-1]) / self.dt))
            new_timestamps = self.time_stamps[-1] + self.dt + np.linspace(0., num_steps * self.dt, num_steps + 1)
            new_epsilons = self.compute_epsilons(num_steps=num_steps + 1)
            new_base_vals = self.base_values[-1] + np.cumsum(new_epsilons)
            self.time_stamps = np.append(self.time_stamps, new_timestamps)
            self.epsilons = np.hstack((self.epsilons, new_epsilons))
            self.base_values = np.hstack((self.base_values, new_base_vals))

        return self.base_values[np.searchsorted(self.time_stamps, t, side='right') - 1].squeeze()
