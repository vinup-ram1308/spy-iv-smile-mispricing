"""
Control test: does the smile-deviation signal actually add value, or is the
backtest result in backtest.py just capturing the volatility risk premium
(the well-documented tendency for implied vol to run above realized vol)?

Method: compute the same simulated PnL for ALL contracts as if you'd sold
every single one (the "naive sell everything" baseline), then compare the
FLAGGED group's PnL against the baseline population via a two-sample
t-test. If flagged trades aren't significantly better than the baseline,
the flagging step isn't adding predictive value.
"""

import numpy as np
import pandas as pd
from scipy import stats
from backtest import payoff_at_expiry


def naive_sell_all_pnl(flagged_df, realized_vol=0.18, n_scenarios=2000, seed=7):
    """Simulated PnL from selling EVERY contract, regardless of flag status."""
    rng = np.random.default_rng(seed)
    pnls = []
    for _, row in flagged_df.iterrows():
        S, T, K = row["S"], row["T"], row["strike"]
        Z = rng.standard_normal(n_scenarios)
        S_T = S * np.exp((0 - 0.5 * realized_vol**2) * T + realized_vol * np.sqrt(T) * Z)
        payoffs = np.array([payoff_at_expiry(s, K, row["option_type"]) for s in S_T])
        pnls.append(row["market_price"] - payoffs.mean())  # PnL from selling
    out = flagged_df.copy()
    out["naive_sell_pnl"] = pnls
    return out


def compare_signal_to_baseline(flagged_df, trades_from_backtest):
    """
    Two-sample t-test: is the signal strategy's PnL significantly different
    from the naive 'sell everything' baseline's PnL?
    """
    all_with_naive = naive_sell_all_pnl(flagged_df)
    baseline_pnl = all_with_naive["naive_sell_pnl"]
    signal_pnl = trades_from_backtest["pnl"]

    t_stat, p_value = stats.ttest_ind(signal_pnl, baseline_pnl, equal_var=False)

    print(f"Baseline (sell everything, n={len(baseline_pnl)}): "
          f"mean={baseline_pnl.mean():.4f}, std={baseline_pnl.std(ddof=1):.4f}")
    print(f"Signal (flagged trades only, n={len(signal_pnl)}): "
          f"mean={signal_pnl.mean():.4f}, std={signal_pnl.std(ddof=1):.4f}")
    print(f"\nDoes the signal beat the baseline? t={t_stat:.4f}, p={p_value:.4f}")
    print("--> Signal adds value beyond the vol risk premium" if p_value < 0.05
          else "--> Signal is NOT distinguishable from the general vol risk premium")

    return {"baseline_mean": baseline_pnl.mean(), "signal_mean": signal_pnl.mean(),
            "t_stat": t_stat, "p_value": p_value}


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_chain
    from analysis import add_implied_vols, fit_smile_and_flag
    from backtest import run_backtest

    raw = generate_synthetic_chain()
    flagged = fit_smile_and_flag(add_implied_vols(raw))
    trades, _ = run_backtest(flagged)
    compare_signal_to_baseline(flagged, trades)
