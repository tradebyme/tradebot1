import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

def fetch_data(symbol="GBPUSD", bars=100000):
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def test_chandelier_and_be(symbol="GBPUSD", bars=100000):
    df = fetch_data(symbol, bars)
    n = len(df)
    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    # ATR 14
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().values

    # EMAs
    ema20 = df["close"].ewm(span=20, adjust=False).mean().values
    ema50 = df["close"].ewm(span=50, adjust=False).mean().values
    ema100 = df["close"].ewm(span=100, adjust=False).mean().values

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).values

    hours = times.dt.hour.values
    session_ok = (hours >= 7) & (hours < 17)

    # Test parameter variations:
    # (be_trigger_r, trail_trigger_r, trail_atr_mult, allow_reentry)
    configs = [
        ("1. Baseline Strategy 1 (Fixed 1.5 ATR SL, 3.0 ATR TP)", None, None, None, False),
        ("2. BE Only: Move SL to +0.2R when MFE >= +1.5R", 1.5, None, None, False),
        ("3. BE @ +1.5R + Chandelier Trail (1.2 ATR) after +2.0R", 1.5, 2.0, 1.2, False),
        ("4. BE @ +1.5R + Chandelier Trail (1.0 ATR) after +1.8R + Reentry", 1.5, 1.8, 1.0, True),
        ("5. BE @ +1.2R + Chandelier Trail (1.2 ATR) after +1.5R + Reentry", 1.2, 1.5, 1.2, True),
        ("6. Chandelier Trail Only (1.2 ATR trail once in profit > 1.2R)", None, 1.2, 1.2, False),
    ]

    print(f"\n==================== [ {symbol} ] INSTITUTIONAL EXIT MATRIX (100k Bars, ~4 Years) ====================")

    for label, be_trig, trail_trig, trail_mult, allow_reentry in configs:
        trades = []
        active_trade = None
        last_exit_dir = None
        last_exit_bar = -999
        reentry_count = 0

        for i in range(120, n):
            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            # 1. Manage Active Trade
            if active_trade is not None:
                tdir = active_trade["dir"]
                entry = active_trade["entry"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                r_dist = active_trade["r_dist"]
                peak_p = active_trade["peak_p"]
                be_done = active_trade["be_done"]

                if tdir == "BUY":
                    if high[i] > peak_p:
                        peak_p = high[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (peak_p - entry) / r_dist

                    # Check Breakeven Shift
                    if (be_trig is not None) and (not be_done) and (curr_mfe_r >= be_trig):
                        new_sl = entry + (0.2 * r_dist) # Entry + 0.2R buffer
                        if new_sl > sl:
                            sl = new_sl
                            active_trade["sl"] = sl
                            active_trade["be_done"] = True

                    # Check Chandelier Trailing Stop
                    if (trail_trig is not None) and (curr_mfe_r >= trail_trig):
                        trail_sl = peak_p - (trail_mult * c_atr)
                        if trail_sl > sl:
                            sl = trail_sl
                            active_trade["sl"] = sl

                    # Check Exits
                    if low[i] <= sl:
                        exit_p = sl
                        pnl = (exit_p - entry) / r_dist * 25.0
                        res = "BE/PROFIT_TRAIL" if pnl >= 0 else "LOSS"
                        trades.append({"pnl": pnl, "res": res})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None
                    elif high[i] >= tp:
                        exit_p = tp
                        pnl = (exit_p - entry) / r_dist * 25.0
                        trades.append({"pnl": pnl, "res": "WIN_TP"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None

                else: # SELL
                    if low[i] < peak_p:
                        peak_p = low[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (entry - peak_p) / r_dist

                    # Check Breakeven Shift
                    if (be_trig is not None) and (not be_done) and (curr_mfe_r >= be_trig):
                        new_sl = entry - (0.2 * r_dist)
                        if new_sl < sl:
                            sl = new_sl
                            active_trade["sl"] = sl
                            active_trade["be_done"] = True

                    # Check Chandelier Trailing Stop
                    if (trail_trig is not None) and (curr_mfe_r >= trail_trig):
                        trail_sl = peak_p + (trail_mult * c_atr)
                        if trail_sl < sl:
                            sl = trail_sl
                            active_trade["sl"] = sl

                    # Check Exits
                    if high[i] >= sl:
                        exit_p = sl
                        pnl = (entry - exit_p) / r_dist * 25.0
                        res = "BE/PROFIT_TRAIL" if pnl >= 0 else "LOSS"
                        trades.append({"pnl": pnl, "res": res})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None
                    elif low[i] <= tp:
                        exit_p = tp
                        pnl = (entry - exit_p) / r_dist * 25.0
                        trades.append({"pnl": pnl, "res": "WIN_TP"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None

            # 2. Check Entries / Re-entries
            if active_trade is None and session_ok[i]:
                buy_sig = False
                sell_sig = False

                # Regular Strategy 1 Signal
                recent_lows = low[i-3:i]
                recent_highs = high[i-3:i]

                is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
                pulled_back_buy = any(l <= (ema20[i] + 0.75 * c_atr) for l in recent_lows)
                rsi_buy = 40.0 <= rsi[i] <= 62.0
                trigger_buy = (close[i] > open_p[i]) and (close[i] > ema20[i])

                is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])
                pulled_back_sell = any(h >= (ema20[i] - 0.75 * c_atr) for h in recent_highs)
                rsi_sell = 38.0 <= rsi[i] <= 60.0
                trigger_sell = (close[i] < open_p[i]) and (close[i] < ema20[i])

                if is_bull_trend and pulled_back_buy and rsi_buy and trigger_buy:
                    buy_sig = True
                elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                    sell_sig = True

                # Re-entry check
                if allow_reentry and (i - last_exit_bar <= 6) and (reentry_count < 1):
                    if last_exit_dir == "BUY" and is_bull_trend and (close[i] > open_p[i]) and (close[i] > ema20[i]):
                        buy_sig = True
                        reentry_count += 1
                    elif last_exit_dir == "SELL" and is_bear_trend and (close[i] < open_p[i]) and (close[i] < ema20[i]):
                        sell_sig = True
                        reentry_count += 1
                else:
                    reentry_count = 0

                if buy_sig:
                    stop_dist = 1.5 * c_atr
                    active_trade = {
                        "dir": "BUY", "entry": close[i], "sl": close[i] - stop_dist,
                        "tp": close[i] + (3.0 * c_atr), "r_dist": stop_dist,
                        "peak_p": close[i], "be_done": False
                    }
                elif sell_sig:
                    stop_dist = 1.5 * c_atr
                    active_trade = {
                        "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                        "tp": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
                        "peak_p": close[i], "be_done": False
                    }

        tdf = pd.DataFrame(trades)
        tot = len(tdf)
        wins = len(tdf[tdf["pnl"] > 0])
        losses = len(tdf[tdf["pnl"] < 0])
        wr = (wins / tot * 100) if tot > 0 else 0
        net = tdf["pnl"].sum() if tot > 0 else 0
        gw = tdf[tdf["pnl"] > 0]["pnl"].sum() if tot > 0 else 0
        gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum()) if tot > 0 else 0
        pf = (gw / gl) if gl > 0 else 999.0

        # Max Drawdown
        cum = tdf["pnl"].cumsum()
        peak = cum.cummax()
        dd = (peak - cum).max()

        print(f" {label}:")
        print(f"   Net PnL: ${net:+8.2f} | PF: {pf:.2f} | Win Rate: {wr:4.1f}% ({wins}W / {losses}L | Total {tot}) | Max DD: -${dd:.2f}")

if __name__ == "__main__":
    for sym in ["GBPUSD", "EURUSD", "XAUUSD"]:
        test_chandelier_and_be(sym)
