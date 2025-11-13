import numpy as np
import pandas as pd
from collections import deque

LOOKBACK = 20
THRESH = 0.10
CASH_ON_NEUTRAL = True

_price_buf = {}               # sym → recent closes
_prev_raw_weights = None      # for the 1-day shift


def _momentum_score(closes):
    if len(closes) < LOOKBACK + 1:
        return np.nan

    s = pd.Series(closes)
    rets = s[:-1].pct_change()             # exclude today's close
    mu  = rets.rolling(LOOKBACK).mean().iloc[-1]
    vol = rets.rolling(LOOKBACK).std().iloc[-1]

    if pd.isna(vol) or vol == 0:
        return np.nan

    return mu / vol


def generate_signals(data_map, config=None):
    global _price_buf, _prev_raw_weights

    # store today's close
    for sym, df in data_map.items():
        px = float(df['close'].iloc[-1])
        if sym not in _price_buf:
            _price_buf[sym] = deque(maxlen=LOOKBACK + 1)
        _price_buf[sym].append(px)

    # compute momentum on rolling window
    raw_signals = {}
    for sym, closes in _price_buf.items():
        sc = _momentum_score(list(closes))
        if pd.isna(sc):
            raw_signals[sym] = 0
        elif sc > THRESH:
            raw_signals[sym] = 1
        else:
            raw_signals[sym] = 0

    # convert long signals to equal weights
    longs = [s for s in raw_signals if raw_signals[s] == 1]

    if len(longs) > 0:
        raw_weights = {s: 1.0 / len(longs) for s in longs}
    else:
        raw_weights = {} if CASH_ON_NEUTRAL else {s: 1.0 / len(raw_signals)}

    # apply 1-day delay (platform requirement)
    tradable = _prev_raw_weights or {}
    _prev_raw_weights = raw_weights

    return tradable
