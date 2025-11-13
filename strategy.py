import numpy as np
import pandas as pd
from collections import deque

LOOKBACK = 20  #for the rolling average we look 20 days before
THRESH = 0.10  #for the score - subject to modification if the signal is greater than 0.1 it is considered strong
CASH_ON_NEUTRAL = True   # if no long and no short : all cash (we dont take any positions)

_price_buf = {}          # dictionnary that score the closing prices for each symbol
_prev_raw_weights = None # store yesterday weights


def _momentum_score(closes):
    """
    close is a list of recent closing price for one symbol
    Risk-adjusted momentum over the most recent LOOKBACK days,
    using data up to *yesterday* to avoid look-ahead.
    """
    if len(closes) < LOOKBACK + 1:
        return np.nan  # when we dont ahve enough data 

    s = pd.Series(closes)
    rets = s[:-1].pct_change()  # exclude today's close
    mu  = rets.rolling(LOOKBACK).mean().iloc[-1] #the most recent rolling average of return (we already removed teh last )
    vol = rets.rolling(LOOKBACK).std().iloc[-1] # same for the volatility

    if pd.isna(vol) or vol == 0:
        return np.nan  # we ensure the data are working

    return mu / vol


def generate_signals(data_map, config=None):
    """
    Long–short momentum:
      - score >  THRESH -> long candidate
      - score < -THRESH -> short candidate
      - else -> flat
    Weights:
      - if both longs & shorts:
          * 50% of capital split equally across longs (positive weights)
          * 50% of capital split equally across shorts (negative weights)
      - if only longs: 100% split across longs
      - if only shorts: 100% split across shorts (net short)
    Then apply 1-day delay: trade using yesterday's raw weights.
    """
    global _price_buf, _prev_raw_weights

    #1 for ecah symbol : 
    # we retrieve the last closing price px of the symbol if the symbol is not in the dictionary we create a key for it and we append the price 
    for sym, df in data_map.items():
        px = float(df["close"].iloc[-1])
        if sym not in _price_buf:
            _price_buf[sym] = deque(maxlen=LOOKBACK + 1)
        _price_buf[sym].append(px)

    # we compute here the raw signal for ach symbol and date
    raw_signals = {} # dictionnary to stock it
    for sym, closes in _price_buf.items():
        sc = _momentum_score(list(closes)) # we retrieve the momentum score for each closing price
        if pd.isna(sc): 
            raw_signals[sym] = 0 # do nothing
        elif sc > THRESH:
            raw_signals[sym] = 1      # long
        elif sc < -THRESH:
            raw_signals[sym] = -1     # short
        else:
            raw_signals[sym] = 0      # flat

    #
    longs  = [s for s, sig in raw_signals.items() if sig == 1]  #list of symbols with tickers +1 fro each day
    shorts = [s for s, sig in raw_signals.items() if sig == -1] #list of tickers with symbol -1

    raw_weights = {}


    # if no assets has any signal stronger than the threshold - either we stay in cash or giv eequal weight to all the portfolio
    if len(longs) == 0 and len(shorts) == 0:
        #keep the cash 
        if CASH_ON_NEUTRAL:
            raw_weights = {}
        else:
            # equal-weight all names (net long)
            n = len(raw_signals)
            if n > 0:
                w = 1.0 / n
                raw_weights = {s: w for s in raw_signals}

    # if we have long and short signals         
    elif len(longs) > 0 and len(shorts) > 0:
        #we split our capital 50-50 between the long and the short signals
        long_w_each  = 0.5 / len(longs)
        short_w_each = -0.5 / len(shorts)
        for s in longs: #we write our weight in a list
            raw_weights[s] = long_w_each
        for s in shorts:
            raw_weights[s] = short_w_each


    elif len(longs) > 0: #if we dont have short signal
        # only longs: 100 split across longs
        long_w_each = 1.0 / len(longs)
        for s in longs:
            raw_weights[s] = long_w_each
    else:
        # only shorts: 100% split across short
        short_w_each = -1.0 / len(shorts)
        for s in shorts:
            raw_weights[s] = short_w_each

    #we shift our sugan
    tradable = _prev_raw_weights or {}   # our weight of yestaerday are tradable but not the one of todayday 1:all cash
    _prev_raw_weights = raw_weights #the previous become the one now

    return tradable
