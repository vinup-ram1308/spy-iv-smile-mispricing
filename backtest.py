"""
Paper backtest: when a strike is flagged 'overpriced', simulate selling it
at market price; when flagged 'underpriced', simulate buying it. Hold to
expiry, settle at intrinsic value, record PnL.

This is deliberately simple — no delta hedging, no transaction costs, no
liquidity constraints, no bid/ask spread modeling. It is NOT a claim that
this is a deployable trading strategy. The point is narrower and more
honest: does the mispricing signal have any real statistical content, or
is it noise? A one-sample t-test on the PnL series answers that directly.
"""

import numpy as np
import pandas as pd
from scipy import stats


def payoff_at_expiry(S_T, K, option_type):
    return max(S_T - K, 0) if option_type == "call" else max(K - S_T, 0)


def run_backtest(flagged_df, S_T_scenarios=None, n_scenarios=2000, seed=7):
    """
    Since this is a synthetic/no-forward-data setup, terminal underlying
    prices are simulated via GBM using each contract's own T and a fixed
    realized-vol assumption, then averaged over n_scenarios paths per
    contract to get an expected PnL. On real data, replace this with the
    ACTUAL realized S_T at expiry for a true backtest.
    """
    rng = np.random.default_rng(seed)
    trades = flagged_df[flagged_df["flagged"]].copy()
    if trades.empty:
        return pd.DataFrame(), None

    realized_vol = 0.18  # assumption, kept separate from priced-in IV on purpose
    pnls = []

    for _, row in trades.iterrows():
        S, T, K = row["S"], row["T"], row["strike"]
        Z = rng.standard_normal(n_scenarios)
        S_T = S * np.exp((0 - 0.5 * realized_vol**2) * T + realized_vol * np.sqrt(T) * Z)
        payoffs = np.array([payoff_at_expiry(s, K, row["option_type"]) for s in S_T])
        expected_payoff = payoffs.mean()

        if row["signal"] == "overpriced":  # sold the option
            pnl = row["market_price"] - expected_payoff
        else:  # underpriced -> bought the option
            pnl = expected_payoff - row["market_price"]

        pnls.append(pnl)

    trades["pnl"] = pnls
    t_stat, p_value = stats.ttest_1samp(trades["pnl"], popmean=0)

    summary = {
        "n_trades": len(trades),
        "mean_pnl": trades["pnl"].mean(),
        "std_pnl": trades["pnl"].std(ddof=1),
        "t_stat": t_stat,
        "p_value": p_value,
    }
    return trades, summary


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_chain
    from analysis import add_implied_vols, fit_smile_and_flag

    raw = generate_synthetic_chain()
    flagged = fit_smile_and_flag(add_implied_vols(raw))
    trades, summary = run_backtest(flagged)

    print(summary)
    print(f"\nSignificant at 5%? {'Yes' if summary['p_value'] < 0.05 else 'No'}")
