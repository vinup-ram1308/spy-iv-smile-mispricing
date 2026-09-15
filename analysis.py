"""
Core research step: for each expiry, back out implied vol per strike from
market prices, fit a smooth smile curve, and flag strikes whose observed IV
deviates from the fitted smile by an unusual amount (z-score of residuals).

This is the actual "quant research" content of the project: turning a raw
options chain into a statistically grounded mispricing signal, rather than
eyeballing it.
"""

import numpy as np
import pandas as pd
from implied_vol import implied_vol


def add_implied_vols(df):
    """Adds an 'iv' column by solving for implied vol on every row."""
    df = df.copy()
    df["iv"] = df.apply(
        lambda row: implied_vol(
            row["market_price"], row["S"], row["strike"], row["T"], row["r"],
            row["option_type"],
        ),
        axis=1,
    )
    return df.dropna(subset=["iv"])


def fit_smile_and_flag(df, z_threshold=1.5):
    """
    Per expiry: fit IV ~ quadratic(log-moneyness) via least squares, compute
    residuals, z-score them, and flag |z| > z_threshold as statistically
    unusual relative to the rest of that expiry's smile.

    Returns df with added columns: log_moneyness, iv_fitted, residual, z,
    flagged, signal ('overpriced'/'underpriced'/None).
    """
    out = []
    for expiry, group in df.groupby("expiry"):
        g = group.copy()
        g["log_moneyness"] = np.log(g["strike"] / g["S"])

        # fit separately isn't necessary — smile is roughly shared across
        # calls/puts at the same strike/expiry since both imply the same
        # risk-neutral density; fit on all rows for that expiry together
        coeffs = np.polyfit(g["log_moneyness"], g["iv"], deg=2)
        g["iv_fitted"] = np.polyval(coeffs, g["log_moneyness"])
        g["residual"] = g["iv"] - g["iv_fitted"]

        resid_std = g["residual"].std(ddof=1)
        g["z"] = g["residual"] / resid_std if resid_std > 0 else 0
        g["flagged"] = g["z"].abs() > z_threshold
        g["signal"] = np.select(
            [g["z"] > z_threshold, g["z"] < -z_threshold],
            ["overpriced", "underpriced"],
            default=None,
        )
        out.append(g)

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_chain

    raw = generate_synthetic_chain()
    with_iv = add_implied_vols(raw)
    flagged = fit_smile_and_flag(with_iv)

    print(f"Total contracts: {len(flagged)}, flagged as mispriced: {flagged['flagged'].sum()}")
    print(flagged[flagged["flagged"]][["expiry", "strike", "option_type", "iv", "iv_fitted", "z", "signal"]])
