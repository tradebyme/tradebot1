import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add both directories to sys.path
sys.path.insert(0, str(Path("e:/OMEN/yaswanth/oanda_not/stategy_1").resolve()))
sys.path.insert(0, str(Path("e:/OMEN/yaswanth/oanda_not/ststegy2").resolve()))

def get_historical_candles(symbol, timeframe="M15", days=730):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    tf = mt5.TIMEFRAME_M15 if timeframe == "M15" else mt5.TIMEFRAME_H1
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def simulate_strategy_1(df, symbol):
    """Strategy 1: Dual EMA (20/50/100) + RSI(14) Pullback + ATR(14) Stop (1:2 R:R)"""
    n = len(df)
    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    # Indicators
    ema20 = df["close"].ewm(span=20, adjust=False).mean().values
    ema50 = df["close"].ewm(span=50, adjust=False).mean().values
    ema100 = df["close"].ewm(span=100, adjust=False).mean().values

    # ATR 14
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().values

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).values

    trades = []
    active_trade = None
    daily_losses = {}

    for i in range(105, n):
        bar_time = times.iloc[i]
        day_str = bar_time.strftime("%Y-%m-%d")
        hour = bar_time.hour
        session_ok = (7 <= hour < 17)

        # Exit check
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
                pnl = 50.0 if exit_res == "WIN" else -25.0
                trades.append({
                    "strategy": "Strategy 1",
                    "symbol": symbol,
                    "entry_time": active_trade["entry_time"],
                    "exit_time": bar_time,
                    "pnl": pnl,
                    "res": exit_res,
                    "day": active_trade["day"]
                })
                if exit_res == "LOSS":
                    t_day = active_trade["day"]
                    daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                active_trade = None

        # Entry check
        if active_trade is None and session_ok:
            if daily_losses.get(day_str, 0) >= 2:
                continue

            c_atr = atr[i]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            recent_lows = low[i-3:i]
            recent_highs = high[i-3:i]

            # Buy check
            is_bull_trend = (ema20[i] > ema50[i]) and (close[i] > ema100[i])
            pulled_back_buy = any(l <= (ema20[i] + 0.75 * c_atr) for l in recent_lows)
            rsi_buy = 40.0 <= rsi[i] <= 62.0
            trigger_buy = (close[i] > open_p[i]) and (close[i] > ema20[i])

            # Sell check
            is_bear_trend = (ema20[i] < ema50[i]) and (close[i] < ema100[i])
            pulled_back_sell = any(h >= (ema20[i] - 0.75 * c_atr) for h in recent_highs)
            rsi_sell = 38.0 <= rsi[i] <= 60.0
            trigger_sell = (close[i] < open_p[i]) and (close[i] < ema20[i])

            if is_bull_trend and pulled_back_buy and rsi_buy and trigger_buy:
                sl = close[i] - (1.5 * c_atr)
                tp = close[i] + (3.0 * c_atr)
                active_trade = {"dir": "BUY", "entry": close[i], "sl": sl, "tp": tp, "day": day_str, "entry_time": bar_time}
            elif is_bear_trend and pulled_back_sell and rsi_sell and trigger_sell:
                sl = close[i] + (1.5 * c_atr)
                tp = close[i] - (3.0 * c_atr)
                active_trade = {"dir": "SELL", "entry": close[i], "sl": sl, "tp": tp, "day": day_str, "entry_time": bar_time}

    return pd.DataFrame(trades)

def simulate_strategy_2(df, symbol):
    """Strategy 2: ALMA(34) Ribbon + Ehlers Adaptive SuperSmoother + 200 EMA Filter + Swing Structure Stops (1:2 R:R)"""
    from strategy import ConfluenceTrendStrategy
    from config import Config

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    # 1. 200 EMA
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values

    # 2. ALMA & StDev
    # Convolution for ALMA
    length = 34
    offset = 0.85
    sigma = 6.0
    m = offset * (length - 1)
    s = length / sigma
    indices = np.arange(length)
    weights = np.exp(-((indices - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()
    alma = np.full(n, np.nan)
    alma[length - 1:] = np.convolve(close, weights[::-1], mode='valid')

    dev = df["close"].rolling(window=34).std().values
    atr = ConfluenceTrendStrategy.calculate_atr(df["high"], df["low"], df["close"], 14).values

    slope_len = 3
    shift_alma = np.roll(alma, slope_len)
    shift_alma[:slope_len] = np.nan
    slope_score = np.where((atr > 0) & (~np.isnan(alma)) & (~np.isnan(shift_alma)),
                           (alma - shift_alma) / atr, 0.0)

    upper_confirm = alma + dev * 0.65
    lower_confirm = alma - dev * 0.65

    bull_setup = (slope_score > 0.08) & (close > upper_confirm)
    bear_setup = (slope_score < -0.08) & (close < lower_confirm)

    ribbon_bull_flip = np.zeros(n, dtype=bool)
    ribbon_bear_flip = np.zeros(n, dtype=bool)
    cur_t = 0
    for i in range(1, n):
        if cur_t != 1 and bull_setup[i]:
            cur_t = 1
            ribbon_bull_flip[i] = True
        elif cur_t != -1 and bear_setup[i]:
            cur_t = -1
            ribbon_bear_flip[i] = True

    # 3. SuperSmoother
    trend_ss, slope, threshold = ConfluenceTrendStrategy.calculate_adaptive_supersmoother(
        df["close"], 14, 50, 9, 3, 10, 0.30, 20, 0.60
    )
    slope = slope.values
    threshold = threshold.values
    ss_long = slope > threshold
    ss_short = slope < -threshold

    ss_regime = np.zeros(n, dtype=int)
    cur_sq = 0
    for i in range(1, n):
        if ss_long[i]:
            cur_sq = 1
        elif ss_short[i]:
            cur_sq = -1
        ss_regime[i] = cur_sq

    # Confluence Triggers with 200 EMA
    buy_mask = ribbon_bull_flip & (ss_regime == 1) & (close > ema200)
    sell_mask = ribbon_bear_flip & (ss_regime == -1) & (close < ema200)

    trades = []
    active_trade = None
    daily_losses = {}
    lookback = 12

    for i in range(250, n):
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
                pnl = 50.0 if exit_res == "WIN" else -25.0
                trades.append({
                    "strategy": "Strategy 2",
                    "symbol": symbol,
                    "entry_time": active_trade["entry_time"],
                    "exit_time": bar_time,
                    "pnl": pnl,
                    "res": exit_res,
                    "day": active_trade["day"]
                })
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

            if buy_mask[i]:
                recent_low = np.min(low[max(0, i - lookback):i])
                raw_risk = close[i] - recent_low
                risk_dist = np.clip(raw_risk, c_atr * 0.75, c_atr * 2.5)
                active_trade = {"dir": "BUY", "entry": close[i], "sl": close[i] - risk_dist, "tp": close[i] + (2.0 * risk_dist), "day": day_str, "entry_time": bar_time}
            elif sell_mask[i]:
                recent_high = np.max(high[max(0, i - lookback):i])
                raw_risk = recent_high - close[i]
                risk_dist = np.clip(raw_risk, c_atr * 0.75, c_atr * 2.5)
                active_trade = {"dir": "SELL", "entry": close[i], "sl": close[i] + risk_dist, "tp": close[i] - (2.0 * risk_dist), "day": day_str, "entry_time": bar_time}

    return pd.DataFrame(trades)

def analyze_performance(trades_df, capital=2000.0):
    tot = len(trades_df)
    if tot == 0:
        return {"trades": 0, "win_rate": 0, "net_pnl": 0, "pf": 0, "max_dd": 0, "max_dd_pct": 0}
    wins = len(trades_df[trades_df["res"] == "WIN"])
    losses = len(trades_df[trades_df["res"] == "LOSS"])
    wr = (wins / tot) * 100
    net = trades_df["pnl"].sum()
    gw = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gl = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    pf = (gw / gl) if gl > 0 else 999.0

    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    trades_df["equity"] = capital + trades_df["cum_pnl"]
    peak = trades_df["equity"].cummax()
    max_dd = (peak - trades_df["equity"]).max()
    max_dd_pct = (max_dd / capital) * 100

    return {
        "trades": tot,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "net_pnl": net,
        "pf": pf,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct
    }

def main():
    symbols = ["GBPUSD", "EURUSD", "XAUUSD", "USDJPY"]
    print("==========================================================================================")
    print("HEAD-TO-HEAD COMPARISON: STRATEGY 1 vs STRATEGY 2 (2 FULL YEARS / ~50,000 CANDLES)")
    print("Rules: Starting Capital $2,000 | Fixed Risk $25 (1.25%) | 1:2 R:R ($50) | Max 2 Daily Losses")
    print("==========================================================================================\n")

    for sym in symbols:
        print(f">>> TESTING {sym} (730 Days M15) <<<")
        df_m15 = get_historical_candles(sym, "M15", 730)
        if df_m15 is None:
            print(f"Skipping {sym}: No data.")
            continue

        res_s1 = simulate_strategy_1(df_m15.copy(), sym)
        res_s2_m15 = simulate_strategy_2(df_m15.copy(), sym)

        df_h1 = get_historical_candles(sym, "H1", 730)
        res_s2_h1 = simulate_strategy_2(df_h1.copy(), sym) if df_h1 is not None else pd.DataFrame()

        m1 = analyze_performance(res_s1)
        m2_m15 = analyze_performance(res_s2_m15)
        m2_h1 = analyze_performance(res_s2_h1)

        print(f"  [STRATEGY 1 (EMA Pullback M15)]       | Trades: {m1['trades']:4d} | WR: {m1['win_rate']:4.1f}% ({m1.get('wins',0)}W/{m1.get('losses',0)}L) | Net: ${m1['net_pnl']:+8.2f} | PF: {m1['pf']:4.2f} | MaxDD: ${m1['max_dd']:6.2f} ({m1['max_dd_pct']:4.1f}%)")
        print(f"  [STRATEGY 2 (ALMA + SS Ribbon M15)]   | Trades: {m2_m15['trades']:4d} | WR: {m2_m15['win_rate']:4.1f}% ({m2_m15.get('wins',0)}W/{m2_m15.get('losses',0)}L) | Net: ${m2_m15['net_pnl']:+8.2f} | PF: {m2_m15['pf']:4.2f} | MaxDD: ${m2_m15['max_dd']:6.2f} ({m2_m15['max_dd_pct']:4.1f}%)")
        print(f"  [STRATEGY 2 (ALMA + SS Ribbon H1)]    | Trades: {m2_h1['trades']:4d} | WR: {m2_h1['win_rate']:4.1f}% ({m2_h1.get('wins',0)}W/{m2_h1.get('losses',0)}L) | Net: ${m2_h1['net_pnl']:+8.2f} | PF: {m2_h1['pf']:4.2f} | MaxDD: ${m2_h1['max_dd']:6.2f} ({m2_h1['max_dd_pct']:4.1f}%)\n")

if __name__ == "__main__":
    main()

