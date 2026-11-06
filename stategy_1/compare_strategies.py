import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

def fetch_mt5_rates(symbol: str, timeframe, days: int = 730):
    mt5.initialize()
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# =====================================================================
# STRATEGY 1: London/NY Momentum Pullback (M15, ATR Stops, Session Filter)
# =====================================================================
def run_strategy_1(df, capital=2000.0, risk_per_trade=25.0):
    # Indicators
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()

    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi_14"] = 100 - (100 / (1 + up / (dn + 1e-9)))

    times = df["time"].dt.to_pydatetime()
    highs, lows, opens, closes = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    atrs, rsis = df["atr_14"].values, df["rsi_14"].values
    e20s, e50s, e100s = df["ema_20"].values, df["ema_50"].values, df["ema_100"].values

    trades = []
    active = None
    last_d = None
    d_losses = 0
    d_pnl = 0.0

    for i in range(100, len(df)):
        bt = times[i]
        b_date = bt.date()
        b_hour = bt.hour
        w = bt.weekday()

        if b_date != last_d:
            last_d = b_date
            d_losses = 0
            d_pnl = 0.0

        if active is not None:
            if active["dir"] == "BUY":
                if lows[i] <= active["sl"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": -risk_per_trade, "result": "LOSS"})
                    d_pnl -= risk_per_trade
                    d_losses += 1
                    active = None
                elif highs[i] >= active["tp"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": risk_per_trade * 2.0, "result": "WIN"})
                    d_pnl += risk_per_trade * 2.0
                    active = None
            elif active["dir"] == "SELL":
                if highs[i] >= active["sl"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": -risk_per_trade, "result": "LOSS"})
                    d_pnl -= risk_per_trade
                    d_losses += 1
                    active = None
                elif lows[i] <= active["tp"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": risk_per_trade * 2.0, "result": "WIN"})
                    d_pnl += risk_per_trade * 2.0
                    active = None

        if active is not None:
            continue

        # Filters: Session (07-17 UTC), weekend, and Max 2 losses daily
        if not (7 <= b_hour < 17) or (w == 4 and b_hour >= 21) or w in (5, 6):
            continue
        if d_losses >= 2:
            continue

        p = i - 1
        atr, rsi, cp, op = atrs[p], rsis[p], closes[p], opens[p]
        e20, e50, e100 = e20s[p], e50s[p], e100s[p]
        if np.isnan(atr) or atr <= 0.0001:
            continue

        r_lows = lows[i-3:i]
        r_highs = highs[i-3:i]

        if (e20 > e50) and (cp > e100) and any(l <= (e20 + 0.75 * atr) for l in r_lows) and (40 <= rsi <= 62) and (cp > op and cp > e20):
            active = {"dir": "BUY", "sl": opens[i] - 1.5 * atr, "tp": opens[i] + 3.0 * atr, "time": bt}
            continue
        if (e20 < e50) and (cp < e100) and any(h >= (e20 - 0.75 * atr) for h in r_highs) and (38 <= rsi <= 60) and (cp < op and cp < e20):
            active = {"dir": "SELL", "sl": opens[i] + 1.5 * atr, "tp": opens[i] - 3.0 * atr, "time": bt}
            continue

    return pd.DataFrame(trades)

# =====================================================================
# STRATEGY 2: EMA_RSI_Trend_EA (12/48 EMA Crossover + RSI 50 Filter)
# =====================================================================
def run_strategy_2(df, capital=2000.0, risk_per_trade=25.0, sl_pips=40, tp_pips=80, spread_pips=2.0):
    # 12 and 48 EMA, RSI 14
    df["f"] = df["close"].ewm(span=12, adjust=False).mean()
    df["s"] = df["close"].ewm(span=48, adjust=False).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["r"] = 100 - (100 / (1 + up / (dn + 1e-9)))

    # Convert pips to price units
    pip_factor = 0.0001
    sl_dist = sl_pips * pip_factor # 0.0040
    tp_dist = tp_pips * pip_factor # 0.0080
    spread_dist = spread_pips * pip_factor # 0.0002

    times = df["time"].dt.to_pydatetime()
    highs, lows, opens, closes = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    f_vals, s_vals, r_vals = df["f"].values, df["s"].values, df["r"].values

    trades = []
    active = None
    last_d = None
    d_pnl = 0.0

    for i in range(50, len(df)):
        bt = times[i]
        b_date = bt.date()
        w = bt.weekday()

        if b_date != last_d:
            last_d = b_date
            d_pnl = 0.0

        if active is not None:
            if active["dir"] == "BUY":
                if lows[i] <= active["sl"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": -risk_per_trade, "result": "LOSS"})
                    d_pnl -= risk_per_trade
                    active = None
                elif highs[i] >= active["tp"]:
                    # 1:2 R:R -> +$50
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": risk_per_trade * 2.0, "result": "WIN"})
                    d_pnl += risk_per_trade * 2.0
                    active = None
            elif active["dir"] == "SELL":
                if highs[i] >= active["sl"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": -risk_per_trade, "result": "LOSS"})
                    d_pnl -= risk_per_trade
                    active = None
                elif lows[i] <= active["tp"]:
                    trades.append({"date": b_date, "entry_time": active["time"], "pnl": risk_per_trade * 2.0, "result": "WIN"})
                    d_pnl += risk_per_trade * 2.0
                    active = None

        if active is not None:
            continue

        # Crossover check on closed bars
        p1 = i - 1
        p2 = i - 2
        f1, f2 = f_vals[p1], f_vals[p2]
        s1, s2 = s_vals[p1], s_vals[p2]
        r1 = r_vals[p1]

        cross_up = (f2 <= s2) and (f1 > s1)
        cross_down = (f2 >= s2) and (f1 < s1)

        if cross_up and r1 > 50.0:
            entry = opens[i] + spread_dist / 2.0
            active = {"dir": "BUY", "sl": entry - sl_dist, "tp": entry + tp_dist, "time": bt}
            continue
        elif cross_down and r1 < 50.0:
            entry = opens[i] - spread_dist / 2.0
            active = {"dir": "SELL", "sl": entry + sl_dist, "tp": entry - tp_dist, "time": bt}
            continue

    return pd.DataFrame(trades)

# =====================================================================
# EVALUATION OF 3-DAY ROLLING WINDOWS
# =====================================================================
def evaluate_3day_challenge(tdf, all_days):
    if len(tdf) == 0:
        return {"passed": 0, "pass_pct": 0, "fail_daily": 0, "daily_pct": 0, "fail_dd": 0, "dd_pct": 0, "neutral": 0}

    total_w = len(all_days) - 2
    pass_cnt, fail_daily, fail_dd, neutral = 0, 0, 0, 0

    for idx in range(total_w):
        w_days = all_days[idx:idx + 3]
        w_t = tdf[tdf["date"].isin(w_days)]
        cum = 0.0
        d_map = {d: 0.0 for d in w_days}
        db, ddb, passed = False, False, False

        for _, r in w_t.iterrows():
            cum += r["pnl"]
            d_map[r["date"]] += r["pnl"]
            if d_map[r["date"]] <= -100.0:
                db = True
            if cum <= -160.0:
                ddb = True
            if cum >= 160.0 and not db and not ddb:
                passed = True

        if passed:
            pass_cnt += 1
        elif db:
            fail_daily += 1
        elif ddb:
            fail_dd += 1
        else:
            neutral += 1

    return {
        "total_windows": total_w,
        "passed": pass_cnt,
        "pass_pct": round((pass_cnt / total_w) * 100, 1),
        "fail_daily": fail_daily,
        "daily_pct": round((fail_daily / total_w) * 100, 1),
        "fail_dd": fail_dd,
        "dd_pct": round((fail_dd / total_w) * 100, 1),
        "neutral": neutral,
        "neutral_pct": round((neutral / total_w) * 100, 1)
    }

def analyze_strategy_results(name, tdf, all_days):
    if len(tdf) == 0:
        print(f"{name:<38} | No trades generated.")
        return

    wins = len(tdf[tdf["result"] == "WIN"])
    losses = len(tdf[tdf["result"] == "LOSS"])
    total = len(tdf)
    win_rate = (wins / total) * 100.0
    net_pnl = tdf["pnl"].sum()
    pf = (wins * 50.0) / (losses * 25.0) if losses > 0 else 999.0

    tdf["cum"] = tdf["pnl"].cumsum()
    max_dd = (tdf["cum"].cummax() - tdf["cum"]).max()

    eval_3d = evaluate_3day_challenge(tdf, all_days)

    print(f"{name:<38} | {total:>5} | {win_rate:>5.1f}% | ${net_pnl:>+9.2f} | {pf:>4.2f} | ${max_dd:>6.2f} | {eval_3d['pass_pct']:>5.1f}% | {eval_3d['daily_pct']:>5.1f}% | {eval_3d['dd_pct']:>5.1f}%")

def main():
    print("=" * 115)
    print("        COMPREHENSIVE MULTI-STRATEGY COMPARISON FOR 3-DAY PROP CHALLENGE        ")
    print("        Dataset: 2 Full Years (Nov 2024 - Nov 2026) directly from MetaTrader 5  ")
    print("        Capital: $2,000 | Target: +8% ($160) | Daily Stop: 5% | Max DD: 8%      ")
    print("=" * 115)
    print(f"{'STRATEGY & TIMEFRAME':<38} | {'TRADES':<5} | {'WIN %':<6} | {'NET PNL':<10} | {'PF':<4} | {'MAX DD':<7} | {'3D WIN':<6} | {'D.FAIL':<6} | {'DD.FAIL'}")
    print("-" * 115)

    # 1. Fetch M15, H1, H4 for GBPUSD & EURUSD
    for sym in ["GBPUSD", "EURUSD"]:
        print(f"\n>>> SYMBOL: {sym}")
        df_m15 = fetch_mt5_rates(sym, mt5.TIMEFRAME_M15, 730)
        df_h1  = fetch_mt5_rates(sym, mt5.TIMEFRAME_H1, 730)
        df_h4  = fetch_mt5_rates(sym, mt5.TIMEFRAME_H4, 730)

        df_m15["date"] = df_m15["time"].dt.date
        all_days = sorted(list(df_m15["date"].unique()))

        # A. Strategy 1: London/NY Momentum Pullback (M15)
        t_strat1 = run_strategy_1(df_m15)
        analyze_strategy_results(f"1. London/NY Pullback (M15)", t_strat1, all_days)

        # B. Strategy 2: EMA_RSI_Trend_EA on H1 (User's suggested timeframe)
        t_strat2_h1 = run_strategy_2(df_h1)
        analyze_strategy_results(f"2. EMA_RSI_Trend_EA (H1)", t_strat2_h1, all_days)

        # C. Strategy 2: EMA_RSI_Trend_EA on H4
        t_strat2_h4 = run_strategy_2(df_h4, sl_pips=80, tp_pips=160) # adjusted for H4 volatility
        analyze_strategy_results(f"3. EMA_RSI_Trend_EA (H4)", t_strat2_h4, all_days)

        # D. Strategy 2: EMA_RSI_Trend_EA on M15
        t_strat2_m15 = run_strategy_2(df_m15, sl_pips=25, tp_pips=50) # adjusted for M15 volatility
        analyze_strategy_results(f"4. EMA_RSI_Trend_EA (M15)", t_strat2_m15, all_days)

    print("=" * 115)

if __name__ == "__main__":
    main()

