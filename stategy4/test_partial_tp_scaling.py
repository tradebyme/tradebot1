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

def test_scaling_models(symbol="GBPUSD", bars=100000):
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
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values

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

    configs = [
        ("1. Baseline Strategy 1 (Fixed 1.5 ATR SL, 3.0 ATR TP, 1:2 R:R)", "baseline", None, None),
        ("2. Partial TP: Close 50% @ +1.5R, Leave 50% to +3.0R (SL at Original SL)", "partial_no_be", 1.5, False),
        ("3. Partial TP: Close 50% @ +1.5R, Move Remaining 50% SL to Breakeven", "partial_be", 1.5, True),
        ("4. Partial TP: Close 50% @ +1.2R, Move Remaining 50% SL to Breakeven", "partial_be_12", 1.2, True),
        ("5. Partial TP: Close 50% @ +1.5R, Trail Remaining 50% with 1.5 ATR Trail", "partial_trail", 1.5, True),
        ("6. Trend Filter (Macro 200 EMA) + Partial TP 50% @ +1.5R + BE on Remaining", "partial_be_macro", 1.5, True),
    ]

    print(f"\n==================== [ {symbol} ] SCALING & RISK MITIGATION MATRIX (100k Bars, ~4 Years) ====================")

    for label, mode, tp1_r, move_be in configs:
        trades = []
        active_trade = None

        for i in range(120, n):
            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            # Manage active trade
            if active_trade is not None:
                tdir = active_trade["dir"]
                entry = active_trade["entry"]
                sl = active_trade["sl"]
                tp2 = active_trade["tp2"]
                r_dist = active_trade["r_dist"]
                partial_done = active_trade["partial_done"]
                partial_pnl = active_trade["partial_pnl"]

                if tdir == "BUY":
                    # Check partial TP
                    if (tp1_r is not None) and (not partial_done) and (high[i] >= entry + (tp1_r * r_dist)):
                        # Bank 50% of the trade at +tp1_r
                        partial_pnl = 0.5 * (tp1_r * 25.0) # $25 risk per trade base
                        partial_done = True
                        active_trade["partial_done"] = True
                        active_trade["partial_pnl"] = partial_pnl
                        if move_be:
                            sl = entry + (0.1 * r_dist) # Move remaining half SL to breakeven + buffer
                            active_trade["sl"] = sl

                    # Check exits for remaining position
                    if low[i] <= sl:
                        remaining_frac = 0.5 if partial_done else 1.0
                        rem_pnl = remaining_frac * ((sl - entry) / r_dist * 25.0)
                        tot_pnl = partial_pnl + rem_pnl
                        res = "WIN_PARTIAL" if tot_pnl > 0 else ("BE" if tot_pnl == 0 else "LOSS")
                        trades.append({"pnl": tot_pnl, "res": res})
                        active_trade = None
                    elif high[i] >= tp2:
                        remaining_frac = 0.5 if partial_done else 1.0
                        rem_pnl = remaining_frac * 50.0 # 2.0R = $50
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN_FULL"})
                        active_trade = None

                else: # SELL
                    if (tp1_r is not None) and (not partial_done) and (low[i] <= entry - (tp1_r * r_dist)):
                        partial_pnl = 0.5 * (tp1_r * 25.0)
                        partial_done = True
                        active_trade["partial_done"] = True
                        active_trade["partial_pnl"] = partial_pnl
                        if move_be:
                            sl = entry - (0.1 * r_dist)
                            active_trade["sl"] = sl

                    if high[i] >= sl:
                        remaining_frac = 0.5 if partial_done else 1.0
                        rem_pnl = remaining_frac * ((entry - sl) / r_dist * 25.0)
                        tot_pnl = partial_pnl + rem_pnl
                        res = "WIN_PARTIAL" if tot_pnl > 0 else ("BE" if tot_pnl == 0 else "LOSS")
                        trades.append({"pnl": tot_pnl, "res": res})
                        active_trade = None
                    elif low[i] <= tp2:
                        remaining_frac = 0.5 if partial_done else 1.0
                        rem_pnl = remaining_frac * 50.0
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN_FULL"})
                        active_trade = None

            # Check new entries
            if active_trade is None and session_ok[i]:
                is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
                is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])

                if mode == "partial_be_macro":
                    is_bull_trend = is_bull_trend and (close[i] > ema200[i])
                    is_bear_trend = is_bear_trend and (close[i] < ema200[i])

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
                        "tp2": close[i] + (3.0 * c_atr), "r_dist": stop_dist,
                        "partial_done": False, "partial_pnl": 0.0
                    }
                elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                    stop_dist = 1.5 * c_atr
                    active_trade = {
                        "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                        "tp2": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
                        "partial_done": False, "partial_pnl": 0.0
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
    for sym in ["GBPUSD", "EURUSD", "XAUUSD"]:
        test_scaling_models(sym)
