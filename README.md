# SPY Options: Implied Volatility Smile Mispricing Signal

A small research project built to explore one question: **can you flag
options that look statistically mispriced relative to their own volatility
smile, and does that signal survive a basic significance test?**

## What it does

1. **`black_scholes.py`** — Black-Scholes pricer and Greeks (delta, gamma,
   vega, theta, rho), implemented from first principles and checked against
   a known textbook value.
2. **`implied_vol.py`** — solves for implied volatility from a market price
   via Brent's method, with no-arbitrage bounds checked before solving.
3. **`data_fetch.py`** — pulls a real SPY option chain via `yfinance`.
4. **`analysis.py`** — for each expiry, backs out IV per strike, fits a
   quadratic smile curve (IV as a function of log-moneyness), and flags any
   strike whose IV deviates from the fitted smile by more than 1.5 standard
   deviations of that expiry's residuals.
5. **`backtest.py`** — for every flagged contract, simulates buying the
   "underpriced" ones and selling the "overpriced" ones, holds to expiry,
   and runs a one-sample t-test on the resulting PnL to check whether it's
   distinguishable from zero.
6. **`main.py`** — runs the full pipeline and produces `iv_smiles.png` and
   `pnl_distribution.png`.

## A note on the data

This was built in an environment without live market access, so the
pipeline was developed and tested against `synthetic_data.py` — a
generated SPY-like chain with a realistic downside-skewed smile and a
handful of intentionally injected mispricings (so the signal-detection
step could be validated: it does correctly recover injected mispricings in
testing).

To run it on real data instead:

```python
# in main.py, change:
USE_REAL_DATA = True
# and swap the data source:
from data_fetch import get_spy_option_chains
raw = get_spy_option_chains()
```

`data_fetch.py` itself hasn't been run end-to-end (same reason), but it's
a standard `yfinance` options-chain pull — it should work as-is with
internet access; worth double-checking column names against whatever
`yfinance` version you have installed.

## Results on the synthetic chain

- 99 contracts analyzed across 4 expiries, 15 flagged as statistically
  mispriced (|z| > 1.5 vs. that expiry's fitted smile).
- Backtest: mean PnL per flagged trade was positive but **not**
  statistically significant at 5% (p ≈ 0.10, n = 15).

That null result is actually the honest and correct one to report here — a
sample of 15 trades is nowhere near enough power to distinguish a real
edge from noise, and this backtest ignores transaction costs, bid/ask
spread, and delta hedging entirely. The right next step, on real data,
would be running this across many more expiries/dates to get a much
larger trade sample before drawing any conclusion about whether the
signal is real.

## Honest limitations

- No delta hedging — PnL is exposed to the underlying's direction, not
  just the vol view, which is not how a real vol-arb desk would trade this.
- No transaction costs or bid/ask spread modeling.
- The "realized vol" used to simulate terminal payoffs is a fixed
  assumption (18%), not calibrated to anything.
- Smile fit is a simple quadratic in log-moneyness — a real desk would use
  something like SVI or a local vol surface.
- This is a research/learning exercise, not a trading strategy pitch.

## Requirements

```
pip install numpy scipy pandas matplotlib yfinance
```
