import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import cpz
from cpz.execution.models import OrderSubmitRequest
from cpz.execution.enums import OrderSide, OrderType, TimeInForce

# --- Set CPZ AI credentials (REPLACE WITH YOUR ACTUAL CREDENTIALS) ---
os.environ["CPZ_AI_API_KEY"] = "your_actual_cpz_api_key_here"  # ← REPLACE THIS
os.environ["CPZ_AI_API_SECRET"] = "your_actual_cpz_secret_here"  # ← REPLACE THIS  
os.environ["CPZ_STRATEGY_ID"] = "3080ca94-7598-46fc-9ea3-1159c46362e8"  # Your actual strategy ID

# ==========================
# Strategy Parameters
# ==========================
LOOKBACK_PERIOD = 20
MOMENTUM_THRESHOLD = 0.10
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
HISTORY_DAYS = 365
PORTFOLIO_VALUE = 100000  # Assume $100,000 portfolio

# ==========================
# CPZ AI / Execution Config
# ==========================
PLACE_ORDERS = False  # <<< SET TO True TO SEND REAL ORDERS
BROKER = "alpaca"
BROKER_ENVIRONMENT = "paper"  # "paper" or "live"

STRATEGY_ID = os.environ.get("CPZ_STRATEGY_ID", "3080ca94-7598-46fc-9ea3-1159c46362e8")


def calculate_momentum_score(prices: pd.Series, lookback: int = LOOKBACK_PERIOD) -> pd.Series:
    """Return a series of risk-adjusted momentum scores (rolling mean / rolling std)."""
    returns = prices.pct_change()
    rolling_mean = returns.rolling(window=lookback).mean()
    rolling_std = returns.rolling(window=lookback).std()
    score = rolling_mean / rolling_std.replace(0, np.nan)
    return score


def fetch_prices(tickers, period_days=HISTORY_DAYS):
    """Fetch adjusted close prices for tickers using yfinance."""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        group_by='ticker',
        threads=True,
    )
    prices = {}
    current_prices = {}

    if isinstance(tickers, str) or len(tickers) == 1:
        t = tickers if isinstance(tickers, str) else tickers[0]
        if 'Adj Close' in data.columns:
            prices[t] = data['Adj Close'].dropna()
            current_prices[t] = data['Adj Close'].iloc[-1]
        elif 'Close' in data.columns:
            prices[t] = data['Close'].dropna()
            current_prices[t] = data['Close'].iloc[-1]
    else:
        for t in tickers:
            try:
                if hasattr(data.columns, "levels") and t in data.columns.levels[0]:
                    ser = data[t].get('Adj Close') if 'Adj Close' in data[t].columns else data[t].get('Close')
                    if ser is not None:
                        prices[t] = ser.dropna()
                        current_prices[t] = ser.iloc[-1]
            except Exception:
                continue

    return prices, current_prices


def momentum_strategy(
    tickers=TICKERS,
    lookback=LOOKBACK_PERIOD,
    threshold=MOMENTUM_THRESHOLD,
    portfolio_value=PORTFOLIO_VALUE,
    place_orders: bool = PLACE_ORDERS,
    strategy_id: str = STRATEGY_ID,
):
    print("=== Momentum Alpha Strategy with Short Selling ===")
    print(f"Portfolio Value: ${portfolio_value:,.2f}\n")
    print(f"Strategy ID: {strategy_id}")
    print(f"Order Placement: {'ENABLED' if place_orders else 'DISABLED (Simulation Mode)'}")
    print(f"Broker: {BROKER} ({BROKER_ENVIRONMENT})\n")

    prices_map, current_prices = fetch_prices(tickers, HISTORY_DAYS)

    signals = {}

    # Calculate signals for all tickers
    for t in tickers:
        if t not in prices_map or len(prices_map[t]) < lookback + 1:
            print(f"{t}: insufficient data. Skipping.")
            signals[t] = {"signal": 0, "momentum": np.nan}
            continue

        series = prices_map[t].sort_index()
        score_series = calculate_momentum_score(series, lookback=lookback)
        current_score = score_series.iloc[-1]

        # Determine signal
        if pd.isna(current_score):
            signal = 0
        elif current_score > threshold:
            signal = 1  # Buy - positive momentum
        elif current_score < -threshold:
            signal = -1  # Short sell - negative momentum
        else:
            signal = 0

        signals[t] = {
            "signal": int(signal),
            "momentum": float(current_score) if not pd.isna(current_score) else np.nan,
        }

    # Portfolio allocation logic
    long_positions = [t for t in tickers if signals[t]["signal"] == 1]
    short_positions = [t for t in tickers if signals[t]["signal"] == -1]
    active_positions = long_positions + short_positions

    n_active = len(active_positions)

    # Equal allocation across active positions
    portfolio_allocation = {}

    if n_active > 0:
        n_absolute_positions = len(long_positions) + len(short_positions)
        allocation_per_position = 1.0 / n_absolute_positions if n_absolute_positions > 0 else 0

        for t in tickers:
            signal = signals[t]["signal"]
            current_price = current_prices.get(t, 0)

            if signal == 1:  # Long
                dollar_allocation = allocation_per_position * portfolio_value
                shares = dollar_allocation / current_price if current_price > 0 else 0

                portfolio_allocation[t] = {
                    'allocation_pct': allocation_per_position * 100,
                    'dollar_amount': dollar_allocation,
                    'shares': shares,
                    'current_price': current_price,
                    'signal': 1,
                    'momentum': signals[t]['momentum'],
                    'position_type': 'LONG',
                }
            elif signal == -1:  # Short
                dollar_allocation = -allocation_per_position * portfolio_value
                shares = dollar_allocation / current_price if current_price > 0 else 0

                portfolio_allocation[t] = {
                    'allocation_pct': -allocation_per_position * 100,
                    'dollar_amount': dollar_allocation,
                    'shares': shares,
                    'current_price': current_price,
                    'signal': -1,
                    'momentum': signals[t]['momentum'],
                    'position_type': 'SHORT',
                }
            else:  # No position
                portfolio_allocation[t] = {
                    'allocation_pct': 0.0,
                    'dollar_amount': 0.0,
                    'shares': 0,
                    'current_price': current_price,
                    'signal': 0,
                    'momentum': signals[t]['momentum'],
                    'position_type': 'CASH',
                }
    else:
        # No active positions - all cash
        for t in tickers:
            portfolio_allocation[t] = {
                'allocation_pct': 0.0,
                'dollar_amount': 0.0,
                'shares': 0,
                'current_price': current_prices.get(t, 0),
                'signal': 0,
                'momentum': signals.get(t, {}).get('momentum', np.nan),
                'position_type': 'CASH',
            }

    # Display results
    print("\n=== Portfolio Allocation ===")
    total_long = 0
    total_short = 0
    total_shares_long = 0
    total_shares_short = 0

    for t in tickers:
        alloc = portfolio_allocation[t]
        position_value = abs(alloc['dollar_amount'])

        if alloc['signal'] == 1:  # Long
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Momentum = {alloc['momentum']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )
            total_long += alloc['allocation_pct']
            total_shares_long += alloc['shares']

        elif alloc['signal'] == -1:  # Short
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Momentum = {alloc['momentum']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )
            total_short += alloc['allocation_pct']
            total_shares_short += abs(alloc['shares'])

        else:  # Cash
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Momentum = {alloc['momentum']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )

    print(f"\n=== Portfolio Summary ===")
    print(f"Long Positions: {len(long_positions)} | Total Long: {total_long:.1f}% | Total Shares: {total_shares_long:.0f}")
    print(f"Short Positions: {len(short_positions)} | Total Short: {total_short:.1f}% | Total Shares: {total_shares_short:.0f}")
    print(f"Net Exposure: {total_long + total_short:.1f}%")
    print(f"Gross Exposure: {total_long + abs(total_short):.1f}%")
    print(f"Cash: {100 - (total_long + abs(total_short)):.1f}%")

    total_invested = sum(abs(alloc['dollar_amount']) for alloc in portfolio_allocation.values())
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Remaining Cash: ${portfolio_value - total_invested:,.2f}")

    # ==========================
    # CPZ Order Placement
    # ==========================
    orders = []
    if place_orders:
        if not strategy_id:
            print("ERROR: STRATEGY_ID is not set. Aborting order placement.")
            sys.exit(1)

        try:
            # Initialize CPZ client
            client = cpz.clients.sync.CPZClient()
            
            # Setup broker
            client.execution.use_broker(BROKER, environment=BROKER_ENVIRONMENT)

            print("\n=== Submitting Orders via CPZ AI ===")
            for symbol, alloc in portfolio_allocation.items():
                signal = alloc["signal"]
                shares = alloc["shares"]

                if signal == 0 or shares == 0:
                    continue

                qty = int(abs(round(shares)))
                if qty <= 0:
                    continue

                side = OrderSide.BUY if signal == 1 else OrderSide.SELL

                order = client.execution.submit_order(OrderSubmitRequest(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=strategy_id,
                ))

                print(f"✅ {side.name} {qty} {symbol} -> Order ID: {order.id}, Status: {order.status}")
                orders.append(order)

            if not orders:
                print("No orders submitted (no active signals).")
            else:
                print(f"\n🎯 Successfully placed {len(orders)} orders!")
                
        except Exception as e:
            print(f"❌ Error during order placement: {e}")
            print("Please check your CPZ AI credentials and internet connection.")

    return {
        "portfolio_allocation": portfolio_allocation,
        "long_positions": long_positions,
        "short_positions": short_positions,
        "net_exposure_pct": total_long + total_short,
        "gross_exposure_pct": total_long + abs(total_short),
        "total_portfolio_value": portfolio_value,
        "orders": orders,
    }


if __name__ == "__main__":
    # Test in simulation mode first, then set PLACE_ORDERS = True
    results = momentum_strategy(portfolio_value=100000)
    
    if not PLACE_ORDERS:
        print(f"\n💡 To enable real trading, set PLACE_ORDERS = True at the top of the file")