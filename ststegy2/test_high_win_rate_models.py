import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

def get_m15_data(symbol="GBPUSD", days=730):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def test_strategy_1_rr(symbol="GBPUSD"):
    df = get_m15_data(symbol, 730)
    n = len(df)
    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    ema20 = df["close"].ewm(span=20, adjust=False).mean().values
    ema50 = df["close"].ewm(span=50, adjust=False).mean().values
    ema100 = df["close"].ewm(span=100, adjust=False).mean().values

    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().values

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).values

    # Test different Risk:Reward ratios
    for rr in [1.0, 1.2, 1.5, 2.0]:
        trades = []
        active_trade = None
        daily_losses = {}

        for i in range(105, n):
            bar_time = times.iloc[i]
            day_str = bar_time.strftime("%Y-%m-%d")
            hour = bar_time.hour
            session_ok = (7 <= hour < 17)

            if active_trade is not None:
                tdir = active_trade["dir"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                exit_res = None

                if tdir == "BUY":
                    if low[i] <= sl:
                        exit_res = "LOSS"
                    elif high[i] >= tp:
                        exit_res = "WIN"
                else:
                    if high[i] >= sl:
                        exit_res = "LOSS"
                    elif low[i] <= tp:
                        exit_res = "WIN"

                if exit_res:
                    pnl = (25.0 * rr) if exit_res == "WIN" else -25.0
                    trades.append({"pnl": pnl, "res": exit_res})
                    if exit_res == "LOSS":
                        t_day = active_trade["day"]
                        daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                    active_trade = None

            if active_trade is None and session_ok:
                if daily_losses.get(day_str, 0) >= 2:
                    continue

                c_atr = atr[i]
                if np.isnan(c_atr) or c_atr <= 0:
                    continue

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
                    sl = close[i] - (1.5 * c_atr)
                    tp = close[i] + (1.5 * c_atr * rr)
                    active_trade = {"dir": "BUY", "entry": close[i], "sl": sl, "tp": tp, "day": day_str}
                elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                    sl = close[i] + (1.5 * c_atr)
                    tp = close[i] - (1.5 * c_atr * rr)
                    active_trade = {"dir": "SELL", "entry": close[i], "sl": sl, "tp": tp, "day": day_str}

        tdf = pd.DataFrame(trades)
        tot = len(tdf)
        wins = len(tdf[tdf["res"] == "WIN"])
        losses = len(tdf[tdf["res"] == "LOSS"])
        wr = (wins / tot) * 100
        net = tdf["pnl"].sum()
        gw = tdf[tdf["pnl"] > 0]["pnl"].sum()
        gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 999.0
        print(f"Strategy 1 on {symbol} (R:R 1:{rr:<3}) | Trades: {tot:4d} | Win Rate: {wr:5.1f}% ({wins}W / {losses}L) | Net PnL: ${net:+8.2f} | PF: {pf:.2f}")

def test_london_sweep(symbol="GBPUSD"):
    """
    London Session Liquidity Sweep (Asian Range Sweep / ICT Judas Swing):
    1. Mark Asian Range: 00:00 - 06:45 UTC High and Low
    2. London Open: 07:00 - 10:00 UTC
    3. If London price pierces Asian High (liquidity sweep), then drops back inside range with bearish candle -> SELL
    4. If London price pierces Asian Low (liquidity sweep), then pops back inside range with bullish candle -> BUY
    5. Target: Opposite side of Asian range or 1:1.5 R:R
    """
    df = get_m15_data(symbol, 730)
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute

    trades = []
    days = df["date"].unique()

    for d in days:
        day_df = df[df["date"] == d].copy().reset_index(drop=True)
        # Asian session: 00:00 to 06:45 UTC
        asian_df = day_df[(day_df["hour"] < 7)]
        if len(asian_df) < 15:
            continue
        asian_high = asian_df["high"].max()
        asian_low = asian_df["low"].min()
        asian_range = asian_high - asian_low

        # If range too tiny or huge, skip
        if asian_range <= 0.0005 or asian_range >= 0.0150:
            continue

        # London window: 07:00 to 11:00 UTC
        london_df = day_df[(day_df["hour"] >= 7) & (day_df["hour"] < 12)]

        entered = False
        for idx, row in london_df.iterrows():
            if entered:
                break
            c = row["close"]
            o = row["open"]
            h = row["high"]
            l = row["low"]

            # Bearish Sweep: Swept Asian High, closed back below Asian High, Red candle
            if h > asian_high and c < asian_high and c < o:
                entry = c
                sl = h + 0.0005 # 5 pips above sweep wick
                risk = sl - entry
                tp = entry - (1.5 * risk) # 1:1.5 target
                entered = True
                direction = "SELL"
                start_i = day_df.index[day_df["time"] == row["time"]][0]

            # Bullish Sweep: Swept Asian Low, closed back above Asian Low, Green candle
            elif l < asian_low and c > asian_low and c > o:
                entry = c
                sl = l - 0.0005 # 5 pips below sweep wick
                risk = entry - sl
                tp = entry + (1.5 * risk)
                entered = True
                direction = "BUY"
                start_i = day_df.index[day_df["time"] == row["time"]][0]

            if entered:
                # Check outcome in remainder of day
                post_df = day_df.iloc[start_i+1:]
                res = None
                for _, p_row in post_df.iterrows():
                    if direction == "BUY":
                        if p_row["low"] <= sl:
                            res = "LOSS"
                            break
                        elif p_row["high"] >= tp:
                            res = "WIN"
                            break
                    else:
                        if p_row["high"] >= sl:
                            res = "LOSS"
                            break
                        elif p_row["low"] <= tp:
                            res = "WIN"
                            break

                # If trade neither hit SL nor TP by end of day, close at market close
                if res is None:
                    last_c = day_df["close"].iloc[-1]
                    if direction == "BUY":
                        pnl = (last_c - entry) / risk * 25.0
                    else:
                        pnl = (entry - last_c) / risk * 25.0
                    res = "WIN" if pnl > 0 else "LOSS"
                    trades.append({"res": res, "pnl": pnl})
                else:
                    pnl = 37.50 if res == "WIN" else -25.0 # 1:1.5
                    trades.append({"res": res, "pnl": pnl})

    tdf = pd.DataFrame(trades)
    tot = len(tdf)
    if tot > 0:
        wins = len(tdf[tdf["res"] == "WIN"])
        losses = len(tdf[tdf["res"] == "LOSS"])
        wr = (wins / tot) * 100
        net = tdf["pnl"].sum()
        gw = tdf[tdf["pnl"] > 0]["pnl"].sum()
        gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else 999.0
        print(f"London Liquidity Sweep on {symbol} (1:1.5 R:R) | Trades: {tot:4d} | Win Rate: {wr:5.1f}% ({wins}W / {losses}L) | Net PnL: ${net:+8.2f} | PF: {pf:.2f}")

if __name__ == "__main__":
    print("Testing Strategy 1 across different R:R ratios on GBPUSD and Gold:")
    test_strategy_1_rr("GBPUSD")
    test_strategy_1_rr("XAUUSD")
    print("\nTesting London Session Liquidity Sweep (Asian Range Sweep / Judas Swing):")
    test_london_sweep("GBPUSD")
    test_london_sweep("EURUSD")

