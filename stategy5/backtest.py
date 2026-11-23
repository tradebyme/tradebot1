"""
Strategy 5 Backtester (OANDA Historical Feed)
=============================================
Quickly backtests the Institutional Trend-ATR Strategy on OANDA historical candles.
Usage:
    python backtest.py --instrument XAU_USD --granularity H4 --count 500
    python backtest.py --instrument EUR_USD --granularity H4 --count 500
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from oanda_api import OandaClient
from strategy import TrendEMAATRStrategy


def run_oanda_backtest(instrument="XAU_USD", granularity="H4", count=500):
    client = OandaClient()
    strategy = TrendEMAATRStrategy()

    print(f"Fetching {count} {granularity} candles for {instrument} from OANDA...")
    raw_candles = client.get_candles(instrument, granularity=granularity, count=count)

    df = strategy.parse_candles(raw_candles)
    if len(df) < 220:
        print(f"Error: Not enough candles received ({len(df)}).")
        return

    data = strategy.compute_indicators(df)

    pip_size, pip_val, _ = client.get_pip_specs(instrument)

    trades = []
    position = None

    for i in range(200, len(data) - 1):
        curr = data.iloc[i]
        prev = data.iloc[i - 1]
        next_bar = data.iloc[i + 1]

        close = curr["close"]
        ema_fast = curr["ema_fast"]
        ema_slow = curr["ema_slow"]
        ema_trend = curr["ema_trend"]
        atr = curr["atr"]

        prev_fast = prev["ema_fast"]
        prev_slow = prev["ema_slow"]

        bull_cross = (prev_fast <= prev_slow) and (ema_fast > ema_slow)
        bear_cross = (prev_fast >= prev_slow) and (ema_fast < ema_slow)

        dist_from_200 = abs(close - ema_trend)
        max_dist = strategy.max_dist_atr_mult * atr

        # Manage open position with trailing stop
        if position:
            high = next_bar["high"]
            low = next_bar["low"]

            if position["direction"] == "LONG":
                # Check initial SL
                if low <= position["sl"]:
                    exit_price = position["sl"]
                    pnl = exit_price - position["entry"]
                    trades.append({"direction": "LONG", "pnl": pnl, "r": pnl / position["risk_dist"]})
                    position = None
                else:
                    # Trailing update
                    if high > position["peak"]:
                        position["peak"] = high
                    if (position["peak"] - position["entry"]) >= Config.TRAIL_ACTIVATION_ATR * position["atr"]:
                        new_sl = position["peak"] - (Config.ATR_TRAIL_MULTIPLIER * position["atr"])
                        if new_sl > position["sl"]:
                            position["sl"] = new_sl

            elif position["direction"] == "SHORT":
                if high >= position["sl"]:
                    exit_price = position["sl"]
                    pnl = position["entry"] - exit_price
                    trades.append({"direction": "SHORT", "pnl": pnl, "r": pnl / position["risk_dist"]})
                    position = None
                else:
                    if low < position["peak"]:
                        position["peak"] = low
                    if (position["entry"] - position["peak"]) >= Config.TRAIL_ACTIVATION_ATR * position["atr"]:
                        new_sl = position["peak"] + (Config.ATR_TRAIL_MULTIPLIER * position["atr"])
                        if new_sl < position["sl"]:
                            position["sl"] = new_sl

        # Look for new signal if no position
        if not position:
            if close > ema_trend and bull_cross and dist_from_200 < max_dist:
                sl_dist = strategy.sl_atr_mult * atr
                position = {
                    "direction": "LONG",
                    "entry": next_bar["open"],
                    "sl": next_bar["open"] - sl_dist,
                    "risk_dist": sl_dist,
                    "peak": next_bar["open"],
                    "atr": atr
                }
            elif close < ema_trend and bear_cross and dist_from_200 < max_dist:
                sl_dist = strategy.sl_atr_mult * atr
                position = {
                    "direction": "SHORT",
                    "entry": next_bar["open"],
                    "sl": next_bar["open"] + sl_dist,
                    "risk_dist": sl_dist,
                    "peak": next_bar["open"],
                    "atr": atr
                }

    if not trades:
        print("No trades triggered in this sample period.")
        return

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    wr = (len(wins) / len(tdf)) * 100
    avg_win = wins["r"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["r"].mean()) if len(losses) > 0 else 1
    payoff = avg_win / avg_loss if avg_loss > 0 else 0
    total_r = tdf["r"].sum()

    print("\n" + "=" * 50)
    print(f" OANDA BACKTEST RESULTS: {instrument} ({granularity})")
    print("=" * 50)
    print(f" Total Trades:    {len(tdf)}")
    print(f" Win Rate:        {wr:.1f}% ({len(wins)} W / {len(losses)} L)")
    print(f" Avg Win / Loss:  {avg_win:+.2f}R / -{avg_loss:.2f}R")
    print(f" Payoff Ratio:    {payoff:.2f}x")
    print(f" Total Return:    {total_r:+.2f}R")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="XAU_USD", help="OANDA Instrument (e.g. XAU_USD, EUR_USD, GBP_USD)")
    parser.add_argument("--granularity", default="H4", help="Candle Granularity (H4, H1)")
    parser.add_argument("--count", type=int, default=500, help="Number of candles")
    args = parser.parse_args()

    run_oanda_backtest(args.instrument, args.granularity, args.count)
