"""
Collection of scripts to simulate stochastic processes
"""
from abc import abstractmethod
from typing import Callable, List, Tuple, Union
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


class GeneralizedRandomWalk():
    """
    Base class for a generalized random walk x(t), where

    dx = drift(x,t) * dt + stochastic_step(x,t)

    Args:
        infinitesimal_time_interval: infinitesimal time interval to be used in simulations
        drift: drift function
        stochastic_stepper: function that computes the stochastic step
    """
    def __init__(self,
                 infinitesimal_time_interval: float,
                 drift: Callable,
                 stochastic_stepper: Callable) -> None:
        self.dt = infinitesimal_time_interval
        self._set_drift(drift=drift)
        self._set_stochastic_stepper(stochastic_stepper=stochastic_stepper)
        self._simulation_history = []

    @property
    def simulation_history(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return deepcopy(self._simulation_history)

    @abstractmethod
    def _set_drift(self, drift: Callable) -> None:
        """
        Auxiliary function to set the drift
        """
        raise ValueError('Please implement in child class!')

    @abstractmethod
    def _set_stochastic_stepper(self, stochastic_stepper: Callable) -> None:
        """
        Auxiliary function to set the diffusion
        """
        raise ValueError('Please implement in child class!')

    @abstractmethod
    def get_drift(self, time: float, x: float) -> Union[float, np.ndarray]:
        """
        Auxiliary function to get drift at given time and given present value of walk
        """
        raise ValueError('Please implement in child class!')

    @abstractmethod
    def get_stochastic_step(self, time: float, x: float) -> Union[float, np.ndarray]:
        """
        Auxiliary function to get diffusion at given time and given present value of walk
        """
        raise ValueError('Please implement in child class!')

    def simulate(self,
                 num_steps: int,
                 start_time: float = 0.0,
                 start_value: Union[float, np.ndarray] = 0.0,
                 save_run: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to simulate the random walk for a given number of steps, given the drift and the diffusion functions.

        Args:
            num_steps: number of steps to simulate
            start_time: starting time of the process
            start_value: starting value of the random variable
            save_run: whether or not to save this run internally
        """
        dt = self.dt
        # Prepare time stamps
        time_stamps = start_time + np.arange(stop=(num_steps * dt), step=dt)
        # Initialize values
        base_values = [start_value]

        # Step
        for t in time_stamps[:-1]:
            drift = self.get_drift(time=t, x=base_values[-1])
            diffusion = self.get_stochastic_step(time=t, x=base_values)
            dx = (drift * dt) + diffusion
            base_values += [base_values[-1] + dx]
        # Convert to numpy array
        base_values = np.array(base_values)

        # save if needed
        if save_run:
            self._simulation_history += [(time_stamps, base_values)]

        return time_stamps, base_values

    def plot_runs(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot all runs. Keyword arguments are passed to `plt.subplots(**kwargs)`
        """
        if len(self._simulation_history) == 0.:
            raise ValueError('No runs have been simulated!')

        fig, axes = plt.subplots(1, 1, **kwargs)
        axes.set_xlabel('Time')
        axes.set_ylabel('Value')

        for time, vals in self._simulation_history:
            axes.plot(time, vals)

        return fig, axes
