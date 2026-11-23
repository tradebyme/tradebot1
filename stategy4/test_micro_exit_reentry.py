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

def test_micro_exit_model(symbol="GBPUSD", bars=50000):
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

    pip_unit = 0.01 if "JPY" in symbol else (0.10 if "XAU" in symbol else 0.0001)
    spread_cost = 1.0 * pip_unit # 1 pip spread per trade entry

    # Models:
    # 1. Baseline: Normal 1.5 ATR SL (-$50), 3.0 ATR TP (+$100)
    # 2. User Idea: If profit touches +$15 (+0.3R), and retraces to -$2 (-0.04R), close! If bounces back on next bar, re-enter!
    # 3. Optimized Model (Strategy 4): Wait for +1.0R (+50) for Breakeven, Trail after +1.2R

    for mode in ["baseline", "user_micro_exit", "strat4_peak_trail"]:
        trades = []
        active_trade = None
        reentry_count = 0
        micro_cut_count = 0

        for i in range(120, n):
            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            # Manage trade
            if active_trade is not None and not active_trade.get("was_cut"):
                tdir = active_trade["dir"]
                entry = active_trade["entry"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                r_dist = active_trade["r_dist"]
                peak_p = active_trade["peak_p"]
                hit_small_profit = active_trade.get("hit_small_profit", False)

                if tdir == "BUY":
                    if high[i] > peak_p:
                        peak_p = high[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (peak_p - entry) / r_dist

                    # User Idea: if reached +0.3R (+$15 on $50 risk)
                    if curr_mfe_r >= 0.30:
                        active_trade["hit_small_profit"] = True
                        hit_small_profit = True

                    # Strategy 4 Peak Trailing
                    if mode == "strat4_peak_trail":
                        if curr_mfe_r >= 1.0 and not active_trade.get("be_done"):
                            sl = entry + (0.1 * r_dist)
                            active_trade["sl"] = sl
                            active_trade["be_done"] = True
                        if curr_mfe_r >= 1.2:
                            trail_sl = peak_p - (0.6 * c_atr)
                            if trail_sl > sl:
                                sl = trail_sl
                                active_trade["sl"] = sl

                    # User Idea: if hit small profit and now dips to -$2 (-0.04R), CUT IMMEDIATELY
                    user_cut_triggered = False
                    if mode == "user_micro_exit" and hit_small_profit:
                        micro_loss_p = entry - (0.04 * r_dist) # -$2 level
                        if low[i] <= micro_loss_p:
                            user_cut_triggered = True

                    if user_cut_triggered:
                        exit_p = entry - (0.04 * r_dist)
                        pnl = -2.0 - (spread_cost / r_dist * 50.0) # -$2 plus spread cost
                        trades.append({"pnl": pnl, "res": "MICRO_CUT"})
                        micro_cut_count += 1
                        active_trade = {"was_cut": True, "dir": "BUY", "cut_bar": i, "entry": entry, "r_dist": r_dist}

                    elif low[i] <= sl:
                        exit_p = sl
                        pnl = (exit_p - entry) / r_dist * 50.0
                        trades.append({"pnl": pnl, "res": "SL"})
                        active_trade = None

                    elif high[i] >= tp:
                        exit_p = tp
                        pnl = 100.0 # 2R
                        trades.append({"pnl": pnl, "res": "TP"})
                        active_trade = None

                else: # SELL
                    if low[i] < peak_p:
                        peak_p = low[i]
                        active_trade["peak_p"] = peak_p

                    curr_mfe_r = (entry - peak_p) / r_dist

                    if curr_mfe_r >= 0.30:
                        active_trade["hit_small_profit"] = True
                        hit_small_profit = True

                    if mode == "strat4_peak_trail":
                        if curr_mfe_r >= 1.0 and not active_trade.get("be_done"):
                            sl = entry - (0.1 * r_dist)
                            active_trade["sl"] = sl
                            active_trade["be_done"] = True
                        if curr_mfe_r >= 1.2:
                            trail_sl = peak_p + (0.6 * c_atr)
                            if trail_sl < sl:
                                sl = trail_sl
                                active_trade["sl"] = sl

                    user_cut_triggered = False
                    if mode == "user_micro_exit" and hit_small_profit:
                        micro_loss_p = entry + (0.04 * r_dist)
                        if high[i] >= micro_loss_p:
                            user_cut_triggered = True

                    if user_cut_triggered:
                        pnl = -2.0 - (spread_cost / r_dist * 50.0)
                        trades.append({"pnl": pnl, "res": "MICRO_CUT"})
                        micro_cut_count += 1
                        active_trade = {"was_cut": True, "dir": "SELL", "cut_bar": i, "entry": entry, "r_dist": r_dist}

                    elif high[i] >= sl:
                        exit_p = sl
                        pnl = (entry - exit_p) / r_dist * 50.0
                        trades.append({"pnl": pnl, "res": "SL"})
                        active_trade = None

                    elif low[i] <= tp:
                        exit_p = tp
                        pnl = 100.0
                        trades.append({"pnl": pnl, "res": "TP"})
                        active_trade = None

            # Check entries / re-entries
            if session_ok[i]:
                is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
                is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])

                # Check if we were micro-cut and should re-enter if recovering above entry
                if active_trade is not None and active_trade.get("was_cut"):
                    cut_dir = active_trade["dir"]
                    cut_bar = active_trade["cut_bar"]
                    if i - cut_bar <= 3:
                        if cut_dir == "BUY" and is_bull_trend and close[i] > open_p[i] and close[i] > active_trade["entry"]:
                            # Re-enter
                            stop_dist = 1.5 * c_atr
                            active_trade = {
                                "dir": "BUY", "entry": close[i], "sl": close[i] - stop_dist,
                                "tp": close[i] + (3.0 * c_atr), "r_dist": stop_dist,
                                "peak_p": close[i], "hit_small_profit": False
                            }
                            reentry_count += 1
                            continue
                        elif cut_dir == "SELL" and is_bear_trend and close[i] < open_p[i] and close[i] < active_trade["entry"]:
                            stop_dist = 1.5 * c_atr
                            active_trade = {
                                "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                                "tp": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
                                "peak_p": close[i], "hit_small_profit": False
                            }
                            reentry_count += 1
                            continue
                    else:
                        active_trade = None

                if active_trade is None:
                    recent_lows = low[i-3:i]
                    recent_highs = high[i-3:i]
                    pulled_back_buy = any(l <= (ema20[i] + 0.75 * c_atr) for l in recent_lows)
                    pulled_back_sell = any(h >= (ema20[i] - 0.75 * c_atr) for h in recent_highs)
                    rsi_buy = 40.0 <= rsi[i] <= 62.0
                    rsi_sell = 38.0 <= rsi[i] <= 60.0
                    trigger_buy = (close[i] > open_p[i]) and (close[i] > ema20[i])
                    trigger_sell = (close[i] < open_p[i]) and (close[i] < ema20[i])

                    if is_bull_trend and pulled_back_buy and rsi_buy and trigger_buy:
                        stop_dist = 1.5 * c_atr
                        active_trade = {
                            "dir": "BUY", "entry": close[i], "sl": close[i] - stop_dist,
                            "tp": close[i] + (3.0 * c_atr), "r_dist": stop_dist,
                            "peak_p": close[i], "hit_small_profit": False
                        }
                    elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                        stop_dist = 1.5 * c_atr
                        active_trade = {
                            "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                            "tp": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
                            "peak_p": close[i], "hit_small_profit": False
                        }

        tdf = pd.DataFrame([t for t in trades if "pnl" in t])
        tot = len(tdf)
        net = tdf["pnl"].sum() if tot > 0 else 0
        wins = len(tdf[tdf["pnl"] > 0])
        losses = len(tdf[tdf["pnl"] < 0])
        wr = (wins / tot * 100) if tot > 0 else 0
        gw = tdf[tdf["pnl"] > 0]["pnl"].sum() if tot > 0 else 0
        gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum()) if tot > 0 else 0
        pf = gw / gl if gl > 0 else 999.0

        cum = tdf["pnl"].cumsum()
        peak = cum.cummax()
        dd = (peak - cum).max()

        print(f"Mode: {mode:<20} | Total Trades: {tot:4d} | Net PnL: ${net:+8.2f} | PF: {pf:.2f} | Win Rate: {wr:4.1f}% | Max DD: -${dd:.2f} | Cuts: {micro_cut_count} | Reentries: {reentry_count}")

if __name__ == "__main__":
    print("Testing on GBPUSD (50,000 Bars, ~2 Years):")
    test_micro_exit_model("GBPUSD")
    print("\nTesting on EURUSD (50,000 Bars, ~2 Years):")
    test_micro_exit_model("EURUSD")
