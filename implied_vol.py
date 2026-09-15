"""
Implied volatility solver: given an observed market option price, find the
sigma that makes Black-Scholes reproduce it. Uses Brent's method (bracketed
root-finding) rather than a black-box IV function, since it's more robust
than Newton-Raphson near-expiry / deep ITM-OTM where vega -> 0.
"""

import numpy as np
from scipy.optimize import brentq
from black_scholes import bs_price


def implied_vol(market_price, S, K, T, r, option_type="call",
                 sigma_lo=1e-4, sigma_hi=5.0):
    """
    Returns implied volatility, or np.nan if no solution exists in the
    bracket (e.g. the market price violates a no-arbitrage bound).
    """
    if T <= 0:
        return np.nan

    # No-arbitrage sanity bounds before we even try to solve
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    upper_bound = S if option_type == "call" else K
    if market_price < intrinsic - 1e-8 or market_price > upper_bound + 1e-8:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, sigma_lo, sigma_hi, xtol=1e-6)
    except ValueError:
        # objective doesn't change sign across the bracket -> no solution found
        return np.nan


if __name__ == "__main__":
    # Round-trip check: price at sigma=0.25, then recover sigma from that price
    true_sigma = 0.25
    price = bs_price(100, 105, 0.5, 0.04, true_sigma, "call")
    recovered = implied_vol(price, 100, 105, 0.5, 0.04, "call")
    print(f"True sigma: {true_sigma}, recovered: {recovered:.6f}")
