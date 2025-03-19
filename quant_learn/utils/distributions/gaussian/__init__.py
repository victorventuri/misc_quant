"""
Collection of scripts to help define basics of Gaussian distribution
"""
from typing import Union

import numpy as np
from scipy.special import erf


def pdf_1d(value: Union[float, np.ndarray],
           mean: Union[float, np.ndarray] = 0.,
           std_dev: Union[float, np.ndarray] = 1.) -> Union[float, np.ndarray]:
    """
    Computes probability density function (PDF) of a Gaussian distribution

    Args:
        value: value (or array of values) for which we wish to know the CDF
        mean: mean of the distribution
        std_dev: standard deviation of the distribution

    Returns:
        CDF value(s) of the desired point(s)
    """
    prefactor = 1./(std_dev * np.sqrt(2 * np.pi))
    exp_factor = - np.power(value - mean, 2) / (2 * (std_dev ** 2))
    return prefactor * np.exp(exp_factor)


def cdf_1d(value: Union[float, np.ndarray],
           mean: Union[float, np.ndarray] = 0.,
           std_dev: Union[float, np.ndarray] = 1.) -> Union[float, np.ndarray]:
    """
    Computes cumulative distribution function (CDF) of a Gaussian distribution

    Args:
        value: value (or array of values) for which we wish to know the CDF
        mean: mean of the distribution
        std_dev: standard deviation of the distribution

    Returns:
        CDF value(s) of the desired point(s)
    """
    adj_val = (value - mean) / (std_dev * np.sqrt(2))
    return 0.5 * (1. + erf(adj_val))
