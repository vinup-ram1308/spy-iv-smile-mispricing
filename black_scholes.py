"""
Black-Scholes option pricing and Greeks, implemented from first principles
(no black-box pricing libraries — only norm.cdf/pdf from scipy for the
standard normal distribution itself).

All functions price European options on a non-dividend-paying underlying.
S     = current underlying price
K     = strike price
T     = time to expiry, in years
r     = risk-free rate (annualized, continuously compounded)
sigma = annualized volatility (decimal, e.g. 0.20 for 20%)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    """European option price under Black-Scholes."""
    if T <= 0 or sigma <= 0:
        # At/after expiry or degenerate vol: payoff is intrinsic value
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Returns delta, gamma, vega, theta, rho.
    Theta is per calendar year (divide by 365 for per-day decay).
    Vega and rho are scaled per 1.00 (100%) move in sigma / r —
    divide by 100 for the conventional 'per 1% move' quote.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    # Sanity check against a well-known textbook value:
    # S=100, K=100, T=1, r=0.05, sigma=0.20 -> call ~= 10.4506
    price = bs_price(100, 100, 1, 0.05, 0.20, "call")
    print(f"ATM call sanity check (expect ~10.45): {price:.4f}")
    print("Greeks:", bs_greeks(100, 100, 1, 0.05, 0.20, "call"))
