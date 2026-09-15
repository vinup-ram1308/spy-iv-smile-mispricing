"""
Generates a synthetic SPY-like options chain with a realistic volatility
smile plus noise, purely so the analysis/backtest pipeline could be built
and tested in an environment without live market data access.

Swap data_fetch.get_spy_option_chains() in for this once you're running
locally with internet access — the rest of the pipeline is agnostic to
where the DataFrame came from, as long as the columns match.
"""

import numpy as np
import pandas as pd
from black_scholes import bs_price

np.random.seed(42)


def true_iv_smile(K, S, T, atm_vol=0.18, skew=-0.5, curvature=1.5):
    """A stylized equity-index smile: downside skew, mild convexity."""
    moneyness = np.log(K / S)
    return max(atm_vol + skew * moneyness + curvature * moneyness**2, 0.03)


def generate_synthetic_chain(S=550.0, r=0.045, n_expiries=4, n_strikes=15):
    rows = []
    expiry_days = [14, 30, 60, 90][:n_expiries]

    for days in expiry_days:
        T = days / 365.0
        strikes = np.linspace(S * 0.85, S * 1.15, n_strikes)

        for K in strikes:
            fair_iv = true_iv_smile(K, S, T)
            # a handful of strikes get an intentional mispricing injected,
            # simulating stale quotes / temporary liquidity gaps
            mispricing_noise = np.random.normal(0, 0.006)  # normal quote noise
            if np.random.rand() < 0.12:
                mispricing_noise += np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
            observed_iv = max(fair_iv + mispricing_noise, 0.01)

            for option_type in ["call", "put"]:
                mkt_price = bs_price(S, K, T, r, observed_iv, option_type)
                mkt_price = max(mkt_price + np.random.normal(0, 0.01), 0.01)  # bid/ask noise
                rows.append({
                    "expiry": f"T+{days}d", "T": T, "strike": K,
                    "option_type": option_type, "market_price": mkt_price,
                    "S": S, "r": r, "volume": np.random.randint(10, 500),
                    "open_interest": np.random.randint(50, 5000),
                    "_true_iv": fair_iv,  # kept only for internal validation, not used by the model
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_chain()
    print(df.shape)
    print(df.head())
    df.to_csv("spy_options_synthetic.csv", index=False)
