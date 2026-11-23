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

def test_alpha(symbol="GBPUSD"):
    df = fetch_data(symbol, 100000)
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

    # Kaufman Efficiency Ratio (10 bars)
    change = (df["close"] - df["close"].shift(10)).abs()
    volatility = (df["close"] - df["close"].shift(1)).abs().rolling(window=10).sum()
    er = (change / (volatility + 1e-9)).fillna(0).values

    # ADX / Trend Strength (14)
    # Directional Movement
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (df["atr"] if "atr" in df else tr.rolling(14).mean() + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (df["atr"] if "atr" in df else tr.rolling(14).mean() + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(14).mean().values

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

    # Let's test combinations of filters:
    experiments = [
        ("Baseline (No Extra Filter, 1:2 R:R)", 0.0, False, False),
        ("Efficiency Filter (ER >= 0.20)", 0.20, False, False),
        ("Efficiency Filter (ER >= 0.25)", 0.25, False, False),
        ("Efficiency Filter (ER >= 0.20) + 200 EMA Macro Alignment", 0.20, True, False),
        ("Efficiency Filter (ER >= 0.20) + ADX >= 20 Trend Strength", 0.20, False, True),
        ("Alpha Stack: ER >= 0.20 + 200 EMA + ADX >= 20", 0.20, True, True),
    ]

    print(f"\n==================== [ {symbol} ] ALPHA FILTER OPTIMIZATION ====================")

    for label, min_er, req_200, req_adx in experiments:
        trades = []
        active_trade = None

        for i in range(120, n):
            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            if active_trade is not None:
                tdir = active_trade["dir"]
                entry = active_trade["entry"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                r_dist = active_trade["r_dist"]
                partial_done = active_trade["partial_done"]
                partial_pnl = active_trade["partial_pnl"]

                if tdir == "BUY":
                    # Partial TP at +1.5R
                    if (not partial_done) and (high[i] >= entry + (1.5 * r_dist)):
                        partial_pnl = 0.5 * (1.5 * 25.0)
                        partial_done = True
                        active_trade["partial_done"] = True
                        active_trade["partial_pnl"] = partial_pnl

                    if low[i] <= sl:
                        rem_frac = 0.5 if partial_done else 1.0
                        rem_pnl = rem_frac * ((sl - entry) / r_dist * 25.0)
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN" if tot_pnl > 0 else "LOSS"})
                        active_trade = None
                    elif high[i] >= tp:
                        rem_frac = 0.5 if partial_done else 1.0
                        rem_pnl = rem_frac * 50.0
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN"})
                        active_trade = None

                else:
                    if (not partial_done) and (low[i] <= entry - (1.5 * r_dist)):
                        partial_pnl = 0.5 * (1.5 * 25.0)
                        partial_done = True
                        active_trade["partial_done"] = True
                        active_trade["partial_pnl"] = partial_pnl

                    if high[i] >= sl:
                        rem_frac = 0.5 if partial_done else 1.0
                        rem_pnl = rem_frac * ((entry - sl) / r_dist * 25.0)
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN" if tot_pnl > 0 else "LOSS"})
                        active_trade = None
                    elif low[i] <= tp:
                        rem_frac = 0.5 if partial_done else 1.0
                        rem_pnl = rem_frac * 50.0
                        tot_pnl = partial_pnl + rem_pnl
                        trades.append({"pnl": tot_pnl, "res": "WIN"})
                        active_trade = None

            if active_trade is None and session_ok[i]:
                # Trend
                is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
                is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])

                if req_200:
                    is_bull_trend = is_bull_trend and (close[i] > ema200[i])
                    is_bear_trend = is_bear_trend and (close[i] < ema200[i])

                if req_adx:
                    is_bull_trend = is_bull_trend and (adx[i] >= 20.0)
                    is_bear_trend = is_bear_trend and (adx[i] >= 20.0)

                if min_er > 0:
                    is_bull_trend = is_bull_trend and (er[i] >= min_er)
                    is_bear_trend = is_bear_trend and (er[i] >= min_er)

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
                        "partial_done": False, "partial_pnl": 0.0
                    }
                elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                    stop_dist = 1.5 * c_atr
                    active_trade = {
                        "dir": "SELL", "entry": close[i], "sl": close[i] + stop_dist,
                        "tp": close[i] - (3.0 * c_atr), "r_dist": stop_dist,
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
    for sym in ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD"]:
        test_alpha(sym)
