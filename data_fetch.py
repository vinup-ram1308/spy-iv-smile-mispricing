"""
Pulls real SPY options chain data via yfinance.

NOTE: this needs an internet connection to Yahoo Finance, which is NOT
available in the sandbox this project was built in — so it hasn't been run
end-to-end here. Run it yourself locally (it's a standard, well-trodden
yfinance call); synthetic_data.py exists purely so the rest of the pipeline
could be built and tested without live data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def get_spy_option_chains(n_expiries=4, r=0.045):
    """
    Returns a single DataFrame with columns:
    expiry, T, strike, option_type, market_price, S, r
    for the first n_expiries available SPY expiries (calls and puts).
    """
    ticker = yf.Ticker("SPY")
    S = ticker.history(period="1d")["Close"].iloc[-1]
    today = datetime.now()

    rows = []
    for expiry in ticker.options[:n_expiries]:
        T = (datetime.strptime(expiry, "%Y-%m-%d") - today).days / 365.0
        if T <= 0:
            continue
        chain = ticker.option_chain(expiry)

        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            for _, row in df.iterrows():
                # mid price is a cleaner signal than last trade price
                if row["bid"] > 0 and row["ask"] > 0:
                    mkt_price = (row["bid"] + row["ask"]) / 2
                else:
                    mkt_price = row["lastPrice"]
                rows.append({
                    "expiry": expiry, "T": T, "strike": row["strike"],
                    "option_type": opt_type, "market_price": mkt_price,
                    "S": S, "r": r, "volume": row.get("volume", np.nan),
                    "open_interest": row.get("openInterest", np.nan),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = get_spy_option_chains()
    print(df.head())
    df.to_csv("spy_options_raw.csv", index=False)
