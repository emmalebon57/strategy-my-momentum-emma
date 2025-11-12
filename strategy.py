# Momentum Alpha Strategy – full backtest + engine outputs
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ===== Parameters =====
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
LOOKBACK = 20
THRESH = 0.10                 # threshold on risk-adjusted momentum
CASH_ON_NEUTRAL = True        # if no signal, hold cash; else equal weight among signaled names
RISKFREE = 0.00               # annual risk-free (for Sharpe)
ANN = 252                     # trading days/year

# ===== Helpers =====
def calc_mom_score(prices: pd.Series, lookback: int) -> pd.Series:
    """Risk-adjusted momentum = mean(returns)/std(returns) over lookback."""
    rets = prices.pct_change()
    mu = rets.rolling(lookback).mean()
    vol = rets.rolling(lookback).std()
    return mu / vol.replace(0, np.nan)

def fetch_prices(tickers, start, end):
    """
    Robust yfinance fetch:
    - auto_adjust=True -> 'Close' is adjusted
    - Handles MultiIndex vs single-index frames
    """
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)  #download the prices from yahoo finance
    # clean the data retrive the right name 
    if isinstance(df.columns, pd.MultiIndex):
        # Standard multi-index: (PriceField, Ticker)
        if 'Close' in df.columns.get_level_values(0):
            data = df['Close']
        elif 'Adj Close' in df.columns.get_level_values(0):
            data = df['Adj Close']
        else:
            raise ValueError("Neither 'Close' nor 'Adj Close' present in yfinance dataframe.")
    else:
        # Single ticker case; normalize to multi-ticker-like frame
        if 'Close' in df.columns:
            data = df[['Close']]
            data.columns = [tickers[0]]
        elif 'Adj Close' in df.columns:
            data = df[['Adj Close']]
            data.columns = [tickers[0]]
        else:
            raise ValueError("Neither 'Close' nor 'Adj Close' present in yfinance dataframe (single ticker).")

    data = data.dropna(how='all')
    # keep only requested tickers (defensive)
    keep = [c for c in data.columns if c in tickers]
    data = data[keep].dropna(how='any')  # align on common business days
    if data.empty:
        raise ValueError("Price data came back empty after cleaning.")
    return data

# ===== Core Backtest =====
def backtest(start=None, end=None, tickers=None):
    if tickers is None:
        tickers = TICKERS
    if end is None:
        end = datetime.now()
    if start is None:
        start = end - timedelta(days=365*2)

    # 1) Prices
    data = fetch_prices(tickers, start, end)

    # 2) Momentum scores
    scores = data.apply(lambda s: calc_mom_score(s, LOOKBACK))

    # 3) Signals: long/flat (set -1 if you want shorting)
    sig_arr = scores.apply(lambda s: np.where(s > THRESH, 1, np.where(s < -THRESH, -1, 0)))
    signals = pd.DataFrame(sig_arr, index=data.index, columns=data.columns).astype(float)

    # Avoid lookahead: use yesterday's signal for today's trade
    signals = signals.shift(1).fillna(0.0)

    # 4) Daily returns
    rets = data.pct_change().fillna(0.0)

    # 5) Weights = equal across active signals each day
    active = (signals != 0).sum(axis=1).replace(0, np.nan)
    weights = signals.div(active, axis=0)
    if CASH_ON_NEUTRAL:
        weights = weights.fillna(0.0)
    else:
        weights = weights.fillna(1.0 / len(tickers))

    # 6) Portfolio daily returns & equity curve
    port_daily = (weights * rets).sum(axis=1)
    equity = (1.0 + port_daily).cumprod()

    # 7) Metrics (printed to logs)
    ytd_mask = equity.index.year == equity.index[-1].year
    if ytd_mask.any() and ytd_mask.sum() > 1:
        ytd_ret = equity[ytd_mask].iloc[-1] / equity[ytd_mask].iloc[0] - 1.0
    else:
        ytd_ret = np.nan

    ann_vol = port_daily.std() * np.sqrt(ANN)
    sharpe = ((port_daily.mean() - RISKFREE/ANN) / (port_daily.std() + 1e-12)) * np.sqrt(ANN)
    max_dd = (equity / equity.cummax() - 1.0).min()

    print("=== Momentum Alpha Strategy – Backtest ===")
    print(f"Period: {equity.index[0].date()} → {equity.index[-1].date()}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"YTD Return: {ytd_ret*100:.2f}% | Ann. Vol: {ann_vol*100:.2f}% | Sharpe: {sharpe:.3f} | Max DD: {max_dd*100:.2f}%")

    # ----- Engine-friendly outputs -----
    out = pd.DataFrame({
        'equity': equity,
        'portfolio_return': port_daily
    })
    out.to_csv('equity_curve.csv', index_label='date')  # for your own 
    

    # ==== Platform adapter (so the runner can call your strategy) ====
# It exposes `generate_signals(data_map, config)` and returns
# today's **tradable** weights = **yesterday's raw equal-weights**.

from collections import deque

# keep a rolling window of closes per symbol to compute the 20-day score
_PRICE_BUF = {}                 # sym -> deque([closes...], maxlen=LOOKBACK+1)
_PREV_RAW_WEIGHTS = None        # yesterday's raw weights (for the 1-day shift)

def _score_last_window(closes):
    """
    Risk-adjusted momentum over the most recent LOOKBACK days,
    using data up to *yesterday* to avoid look-ahead.
    """
    s = pd.Series(closes)
    if len(s) < LOOKBACK + 1:
        return np.nan
    rets = s[:-1].pct_change()                     # exclude today's close
    mu  = rets.rolling(LOOKBACK).mean().iloc[-1]
    vol = rets.rolling(LOOKBACK).std().iloc[-1]
    if pd.isna(vol) or vol == 0:
        return np.nan
    return mu / vol

def generate_signals(data_map, config=None):
    """
    Called by the platform every bar.
    - data_map: {'AAPL': DataFrame({'close':[today_close]}), ...}
    - returns: dict of weights for *today's trades*
               (which are yesterday's raw weights → 1-day shift)
    """
    global _PRICE_BUF, _PREV_RAW_WEIGHTS

    # 1) Update rolling close buffers with today's close
    for sym, df in data_map.items():
        px = float(df['close'].iloc[-1])
        if sym not in _PRICE_BUF:
            _PRICE_BUF[sym] = deque(maxlen=LOOKBACK + 1)
        _PRICE_BUF[sym].append(px)

    # 2) Compute today's RAW signals (+1/0/-1) from yesterday's window
    raw_signals = {}
    for sym, q in _PRICE_BUF.items():
        sc = _score_last_window(list(q))
        if pd.isna(sc):
            raw_signals[sym] = 0
        elif sc > THRESH:
            raw_signals[sym] = 1
        elif sc < -THRESH:
            raw_signals[sym] = -1   # unused if you don't short
        else:
            raw_signals[sym] = 0

    # 3) Convert raw signals to RAW equal-weights (long-only, like your code)
    longs = [s for s, sig in raw_signals.items() if sig == 1]
    if len(longs) > 0:
        raw_weights = {s: 1.0 / len(longs) for s in longs}
    else:
        raw_weights = {} if CASH_ON_NEUTRAL else {s: 1.0 / len(raw_signals) for s in raw_signals}

    # 4) Apply the **1-day shift**: trade using *yesterday's* raw weights
    tradable = _PREV_RAW_WEIGHTS or {}   # empty dict ⇒ all cash today
    _PREV_RAW_WEIGHTS = raw_weights

    return tradable

