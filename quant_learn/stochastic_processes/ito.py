"""
Scripts to simulate Itô process
"""
from typing import Optional, Union, List, Tuple

import numpy as np

from .wiener import GeneralizedWiener as Wiener


class ItoProcess1D():
    """
    Base definition of an Itô process

    Args:
        seed: seed to be used in random number generator
    """
    def __init__(self,
                 seed: int) -> None:
        self.seed = seed

    @property
    def seed(self) -> int:
        return self.wiener.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.wiener = Wiener(seed=value)
