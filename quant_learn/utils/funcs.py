"""
Utility definitions for functions and classes
"""
from typing import Union

import numpy as np


class ConstantFunction():
    """
    Basically a dummy function that always returns a certain constant value, regardless of the inputs.

    Args:
        constant: constant value to return; defaults to 0.0
    """
    def __init__(self, constant: Union[float, np.ndarray] = 0.0):
        self.constant = constant

    def __call__(self, *args, **kwargs) -> Union[float, np.ndarray]:
        return self.constant
