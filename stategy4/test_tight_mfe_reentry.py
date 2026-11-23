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

def test_peak_trailing_reentry(symbol="GBPUSD", bars=100000):
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

    # Test varying trailing pullbacks:
    # (activation_r, pullback_atr_buffer, allow_reentry)
    tests = [
        ("1. Baseline (Fixed 1.5 ATR SL, 3.0 ATR TP)", None, None, False),
        ("2. Peak Trailing: Activate @ +1.2R, Trail by 0.6 ATR, No Reentry", 1.2, 0.6, False),
        ("3. Peak Trailing: Activate @ +1.2R, Trail by 0.6 ATR + Smart Reentry", 1.2, 0.6, True),
        ("4. Peak Trailing: Activate @ +1.5R, Trail by 0.75 ATR, No Reentry", 1.5, 0.75, False),
        ("5. Peak Trailing: Activate @ +1.5R, Trail by 0.75 ATR + Smart Reentry", 1.5, 0.75, True),
        ("6. Peak Trailing: Activate @ +1.5R, Trail by 1.0 ATR + Smart Reentry", 1.5, 1.0, True),
    ]

    print(f"\n==================== [ {symbol} ] PEAK TRAILING & RE-ENTRY EXPERIMENT ====================")

    for label, act_r, trail_buf, allow_reentry in tests:
        trades = []
        active_trade = None
        last_exit_dir = None
        last_exit_bar = -999
        reentry_count = 0

        for i in range(120, n):
            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            # Manage active trade
            if active_trade is not None:
                tdir = active_trade["dir"]
                entry = active_trade["entry"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                r_dist = active_trade["r_dist"]
                peak_p = active_trade["peak_p"]

                if tdir == "BUY":
                    if high[i] > peak_p:
                        peak_p = high[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (peak_p - entry) / r_dist

                    # If activated, trail stop behind peak
                    if (act_r is not None) and (curr_mfe_r >= act_r):
                        trail_sl = peak_p - (trail_buf * c_atr)
                        if trail_sl > sl:
                            sl = trail_sl
                            active_trade["sl"] = sl

                    if low[i] <= sl:
                        exit_p = sl
                        pnl = (exit_p - entry) / r_dist * 25.0
                        trades.append({"pnl": pnl, "res": "WIN" if pnl > 0 else "LOSS"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None
                    elif high[i] >= tp:
                        exit_p = tp
                        pnl = 50.0 # 2R
                        trades.append({"pnl": pnl, "res": "WIN"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None

                else: # SELL
                    if low[i] < peak_p:
                        peak_p = low[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (entry - peak_p) / r_dist

                    if (act_r is not None) and (curr_mfe_r >= act_r):
                        trail_sl = peak_p + (trail_buf * c_atr)
                        if trail_sl < sl:
                            sl = trail_sl
                            active_trade["sl"] = sl

                    if high[i] >= sl:
                        exit_p = sl
                        pnl = (entry - exit_p) / r_dist * 25.0
                        trades.append({"pnl": pnl, "res": "WIN" if pnl > 0 else "LOSS"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None
                    elif low[i] <= tp:
                        exit_p = tp
                        pnl = 50.0
                        trades.append({"pnl": pnl, "res": "WIN"})
                        last_exit_dir = tdir
                        last_exit_bar = i
                        active_trade = None

            # New Entries / Re-entries
            if active_trade is None and session_ok[i]:
                is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
                is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])

                recent_lows = low[i-3:i]
                recent_highs = high[i-3:i]
                pulled_back_buy = any(l <= (ema20[i] + 0.75 * c_atr) for l in recent_lows)
                pulled_back_sell = any(h >= (ema20[i] - 0.75 * c_atr) for h in recent_highs)
                rsi_buy = 40.0 <= rsi[i] <= 62.0
                rsi_sell = 38.0 <= rsi[i] <= 60.0
                trigger_buy = (close[i] > open_p[i]) and (close[i] > ema20[i])
                trigger_sell = (close[i] < open_p[i]) and (close[i] < ema20[i])

                buy_sig = is_bull_trend and pulled_back_buy and rsi_buy and trigger_buy
                sell_sig = is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell

                # Check Smart Re-entry: within 6 bars of previous exit in same direction
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
                        "peak_p": close[i]
                    }
                elif sell_sig:
                    stop_dist = 1.5 * c_atr
                    active_trade = {
                        "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                        "tp": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
                        "peak_p": close[i]
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

        cum = tdf["pnl"].cumsum()
        peak = cum.cummax()
        dd = (peak - cum).max()

        print(f" {label}:")
        print(f"   Net PnL: ${net:+8.2f} | PF: {pf:.2f} | Win Rate: {wr:4.1f}% ({wins}W / {losses}L | Total {tot}) | Max DD: -${dd:.2f}")

if __name__ == "__main__":
    for sym in ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD"]:
        test_peak_trailing_reentry(sym)
