"""
Scripts to compute Black-Scholes valuation of European options
"""
from typing import Optional

import numpy as np

from quant_learn.utils.distributions import gaussian

from . import OptionPrice


def black_scholes_value(strike_price: float,
                        spot_price: float,
                        time_to_maturity: float,
                        interest_rate: float,
                        volatility: float,
                        drift: Optional[float] = None,
                        dividend_yield: float = 0.
                        ) -> OptionPrice:
    """
    Function to compute Black-Scholes valuation of European call and put options.

    Args:
        strike_price: strike price determined by option contract
        spot_price: present price of underlying asset
        time_to_maturity: time to maturity of option contract
        interest_rate: risk-free interest rate, which should be provided in corresponding units of the time_to_maturity
        volatility: volatility of underlying asset
        drift: drift of underlying asset; if not provided, assumed to be equal to interest rate
        divident_yield: continuous dividient yield, if any; defaults to 0.

    Returns:
        dictionary containing Black-Scholes valuation of the call and put option
    """
    # Risk-neutral assumption
    if drift is None:
        drift = interest_rate

    # Other relevant computations
    moneyness = np.log(spot_price / strike_price)
    expected_S_T = spot_price * np.exp((drift - dividend_yield) * time_to_maturity)

    # d-params
    d_prefactor = 1. / (volatility * np.sqrt(time_to_maturity))
    d_plus = time_to_maturity * (drift - dividend_yield + ((volatility ** 2) / 2))
    d_minus = time_to_maturity * (drift - dividend_yield - ((volatility ** 2) / 2))
    d_plus += moneyness
    d_minus += moneyness
    d_plus *= d_prefactor
    d_minus *= d_prefactor

    # Main calculations
    call_value = (expected_S_T * gaussian.cdf_1d(value=d_plus)) - (strike_price * gaussian.cdf_1d(value=d_minus))
    put_value = (strike_price * gaussian.cdf_1d(value=-d_plus)) - (expected_S_T * gaussian.cdf_1d(value=-d_minus))
    # Discount them to present value
    call_value *= np.exp(-interest_rate * time_to_maturity)
    put_value *= np.exp(-interest_rate * time_to_maturity)

    return OptionPrice(call=call_value, put=put_value)
