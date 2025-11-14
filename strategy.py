# momentum_alpha.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Strategy Parameters
LOOKBACK_PERIOD = 20
MOMENTUM_THRESHOLD = 0.10   # same as your 0.1
POSITION_SIZE = 0.10        # fraction used to compute contribution in the simple summary
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
HISTORY_DAYS = 365

def calculate_momentum_score(prices: pd.Series, lookback: int = LOOKBACK_PERIOD) -> pd.Series:
    """Return a series of risk-adjusted momentum scores (rolling mean / rolling std)"""
    returns = prices.pct_change()
    # require at least `lookback` non-null returns for rolling windows, otherwise result is NaN
    rolling_mean = returns.rolling(window=lookback).mean()
    rolling_std  = returns.rolling(window=lookback).std()
    # avoid division by zero
    score = rolling_mean / rolling_std.replace(0, np.nan)
    return score

def fetch_prices(tickers, period_days=HISTORY_DAYS):
    """Fetch adjusted close prices for tickers using yfinance. Returns dict[ticker] -> pd.Series"""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                       progress=False, group_by='ticker', threads=True)
    prices = {}
    # yfinance returns different shapes for single vs multi symbols; handle both
    if isinstance(tickers, str) or len(tickers) == 1:
        # single ticker -> data has 'Adj Close' column
        t = tickers if isinstance(tickers, str) else tickers[0]
        if 'Adj Close' in data.columns:
            prices[t] = data['Adj Close'].dropna()
        elif 'Close' in data.columns:
            prices[t] = data['Close'].dropna()
    else:
        for t in tickers:
            try:
                if (t in data.columns.levels[0]) if hasattr(data.columns, "levels") else False:
                    ser = data[t].get('Adj Close') if 'Adj Close' in data[t].columns else data[t].get('Close')
                    if ser is not None:
                        prices[t] = ser.dropna()
                else:
                    # fallback: try single-column frame (some tickers missing)
                    if 'Adj Close' in data.columns:
                        prices[t] = data['Adj Close'].dropna()
            except Exception:
                continue
    return prices

def momentum_strategy(tickers=TICKERS,
                      lookback=LOOKBACK_PERIOD,
                      threshold=MOMENTUM_THRESHOLD,
                      position_size=POSITION_SIZE):
    print("=== Momentum Alpha Strategy ===")
    prices_map = fetch_prices(tickers, HISTORY_DAYS)

    signals = {}
    contributions = []  # used to compute a simple aggregated "portfolio return" summary

    for t in tickers:
        if t not in prices_map or len(prices_map[t]) < lookback + 1:
            print(f"{t}: insufficient data (got {len(prices_map.get(t, []))} rows). Skipping.")
            signals[t] = {"signal": 0, "momentum": np.nan, "recent_return_pct": np.nan}
            continue

        series = prices_map[t].sort_index()  # ensure chronological order
        score_series = calculate_momentum_score(series, lookback=lookback)
        current_score = score_series.iloc[-1]

        # Determine signal
        if pd.isna(current_score):
            signal = 0
        elif current_score > threshold:
            signal = 1
        elif current_score < -threshold:
            signal = -1
        else:
            signal = 0

        # recent return over last 21 trading days (~1 month)
        lookback_days = 21
        if len(series) > lookback_days:
            recent_return = (series.iloc[-1] / series.iloc[-lookback_days] - 1) * 100
        else:
            recent_return = (series.iloc[-1] / series.iloc[0] - 1) * 100

        # For simple portfolio summary: treat POSITION_SIZE fraction of capital per active position
        contributions.append(recent_return * position_size if signal != 0 else 0)

        signals[t] = {
            "signal": int(signal),
            "momentum": float(current_score) if not pd.isna(current_score) else np.nan,
            "recent_return_pct": float(recent_return)
        }

        print(f"{t}: Momentum Score = {signals[t]['momentum']:.4f} | Signal = {signals[t]['signal']} | "
              f"Recent Return (21d) = {signals[t]['recent_return_pct']:.2f}%")

    # portfolio performance summary (very simple aggregation)
    active_positions = [c for c in contributions if c != 0]
    total_return_est = sum(contributions)
    avg_pos_return = np.mean(active_positions) if active_positions else 0.0

    print("\nPortfolio Performance Summary:")
    print(f" Total (sum of position contributions): {total_return_est:.2f}%")
    print(f" Number of active positions: {len(active_positions)}")
    print(f" Average active position contribution: {avg_pos_return:.2f}%")

    return {"signals": signals,
            "total_return_est_pct": total_return_est,
            "n_active": len(active_positions),
            "avg_active_contrib_pct": avg_pos_return}

if __name__ == "__main__":
    momentum_strategy()
