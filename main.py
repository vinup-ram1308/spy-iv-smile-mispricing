"""
End-to-end pipeline. Uses synthetic_data by default since this project was
built in an environment without live market access — swap in
data_fetch.get_spy_option_chains() for a real run (see README).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import generate_synthetic_chain
from analysis import add_implied_vols, fit_smile_and_flag
from backtest import run_backtest

# swap this line for: from data_fetch import get_spy_option_chains as _get_data
USE_REAL_DATA = False


def main():
    raw = generate_synthetic_chain() if not USE_REAL_DATA else None
    with_iv = add_implied_vols(raw)
    flagged = fit_smile_and_flag(with_iv)
    trades, summary = run_backtest(flagged)

    print("=== Signal summary ===")
    print(f"Contracts analyzed: {len(flagged)}")
    print(f"Flagged as statistically mispriced: {flagged['flagged'].sum()}")
    print("\n=== Backtest summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    plot_smiles(flagged)
    plot_pnl_distribution(trades)


def plot_smiles(flagged):
    expiries = flagged["expiry"].unique()
    fig, axes = plt.subplots(1, len(expiries), figsize=(4.5 * len(expiries), 4), sharey=True)
    if len(expiries) == 1:
        axes = [axes]

    for ax, expiry in zip(axes, expiries):
        g = flagged[flagged["expiry"] == expiry].sort_values("log_moneyness")
        ax.scatter(g["log_moneyness"], g["iv"], s=18, c="steelblue", label="Observed IV")
        ax.plot(g["log_moneyness"], g["iv_fitted"], c="black", lw=1.2, label="Fitted smile")
        flagged_pts = g[g["flagged"]]
        ax.scatter(flagged_pts["log_moneyness"], flagged_pts["iv"], s=55,
                    facecolors="none", edgecolors="red", linewidths=1.6, label="Flagged")
        ax.set_title(expiry)
        ax.set_xlabel("log-moneyness")

    axes[0].set_ylabel("Implied volatility")
    axes[0].legend(fontsize=8)
    fig.suptitle("SPY implied volatility smile by expiry, with flagged mispricings", y=1.03)
    fig.tight_layout()
    fig.savefig("iv_smiles.png", dpi=150, bbox_inches="tight")
    print("\nSaved iv_smiles.png")


def plot_pnl_distribution(trades):
    if trades.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trades["pnl"], bins=10, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(trades["pnl"].mean(), color="red", lw=1.5, linestyle="--", label="Mean PnL")
    ax.set_xlabel("Simulated PnL per flagged trade")
    ax.set_ylabel("Count")
    ax.set_title("Backtest PnL distribution (flagged trades, held to expiry)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pnl_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved pnl_distribution.png")


if __name__ == "__main__":
    main()
