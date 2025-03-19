"""
Collection of scripts pertaining to options
"""
from typing import TypedDict


class OptionPrice(TypedDict):
    """
    Simple container to help specify how to price call and options with the same parameters (time to maturity, strike
    price, interest rate, etc.)
    """
    call: float
    put: float
