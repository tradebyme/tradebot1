import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

def fetch_gbp_data():
    mt5.initialize()
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=730)
    rates = mt5.copy_rates_range("GBPUSD", mt5.TIMEFRAME_M15, utc_from, utc_to)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Base indicators
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()

    # ATR 14
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(window=14).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ADX 14
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = df["tr"].ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    return df

def simulate_strategy(df, move_to_breakeven=False, filter_strong_body=False, filter_adx=False, adx_min=20.0, rr_ratio=2.0):
    times = df["time"].dt.to_pydatetime()
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    atr_vals = df["atr_14"].values
    rsi_vals = df["rsi_14"].values
    ema20_vals = df["ema_20"].values
    ema50_vals = df["ema_50"].values
    ema100_vals = df["ema_100"].values
    adx_vals = df["adx_14"].values

    trades = []
    active_trade = None
    last_trade_day = None
    daily_trades_count = 0
    daily_pnl = 0.0
    circuit_breaker_active = False

    risk_per_trade = 25.0 # $25

    n = len(df)
    for i in range(100, n):
        bar_time = times[i]
        bar_date = bar_time.date()
        bar_hour = bar_time.hour
        weekday = bar_time.weekday()

        if bar_date != last_trade_day:
            last_trade_day = bar_date
            daily_trades_count = 0
            daily_pnl = 0.0
            circuit_breaker_active = False

        # Manage active trade
        if active_trade is not None:
            c_low = lows[i]
            c_high = highs[i]

            # Breakeven adjustment
            if move_to_breakeven and not active_trade.get("be_moved", False):
                if active_trade["direction"] == "BUY":
                    # If high reached +1R (entry + 1.5 ATR)
                    if c_high >= (active_trade["entry_price"] + 1.5 * active_trade["atr"]):
                        active_trade["sl"] = active_trade["entry_price"] # Move SL to Breakeven
                        active_trade["be_moved"] = True
                elif active_trade["direction"] == "SELL":
                    if c_low <= (active_trade["entry_price"] - 1.5 * active_trade["atr"]):
                        active_trade["sl"] = active_trade["entry_price"]
                        active_trade["be_moved"] = True

            # Check SL & TP
            if active_trade["direction"] == "BUY":
                if c_low <= active_trade["sl"]:
                    is_be = active_trade.get("be_moved", False) and active_trade["sl"] >= active_trade["entry_price"]
                    pnl = 0.0 if is_be else -risk_per_trade
                    trades.append({"pnl": pnl, "result": "BE" if is_be else "LOSS", "date": bar_date})
                    daily_pnl += pnl
                    active_trade = None
                elif c_high >= active_trade["tp"]:
                    pnl = risk_per_trade * rr_ratio
                    trades.append({"pnl": pnl, "result": "WIN", "date": bar_date})
                    daily_pnl += pnl
                    active_trade = None

            elif active_trade["direction"] == "SELL":
                if c_high >= active_trade["sl"]:
                    is_be = active_trade.get("be_moved", False) and active_trade["sl"] <= active_trade["entry_price"]
                    pnl = 0.0 if is_be else -risk_per_trade
                    trades.append({"pnl": pnl, "result": "BE" if is_be else "LOSS", "date": bar_date})
                    daily_pnl += pnl
                    active_trade = None
                elif c_low <= active_trade["tp"]:
                    pnl = risk_per_trade * rr_ratio
                    trades.append({"pnl": pnl, "result": "WIN", "date": bar_date})
                    daily_pnl += pnl
                    active_trade = None

            if daily_pnl <= -70.0:
                circuit_breaker_active = True

        if active_trade is not None:
            continue

        if not (7 <= bar_hour < 17) or (weekday == 4 and bar_hour >= 21) or weekday in (5, 6):
            continue
        if circuit_breaker_active or daily_trades_count >= 3:
            continue

        prev_idx = i - 1
        atr = atr_vals[prev_idx]
        rsi = rsi_vals[prev_idx]
        ema20 = ema20_vals[prev_idx]
        ema50 = ema50_vals[prev_idx]
        ema100 = ema100_vals[prev_idx]
        c_p = closes[prev_idx]
        o_p = opens[prev_idx]
        h_p = highs[prev_idx]
        l_p = lows[prev_idx]
        adx = adx_vals[prev_idx]

        if np.isnan(atr) or atr <= 0.0001:
            continue

        # Filter: Strong Body (Rejects weak dojis/indecision bars)
        if filter_strong_body:
            total_range = h_p - l_p
            body = abs(c_p - o_p)
            if total_range > 0 and (body / total_range) < 0.40:
                continue # Skip indecision candles

        # Filter: ADX Trend Strength
        if filter_adx and adx < adx_min:
            continue

        recent_lows = lows[i-3:i]
        recent_highs = highs[i-3:i]

        # BUY Signal
        is_bull_trend = (ema20 > ema50) and (c_p > ema100)
        pulled_back_support = any(l <= (ema20 + 0.75 * atr) for l in recent_lows)
        rsi_bull = 40.0 <= rsi <= 62.0
        trigger_bull = (c_p > o_p) and (c_p > ema20)

        if is_bull_trend and pulled_back_support and rsi_bull and trigger_bull:
            entry = opens[i]
            sl = entry - (1.5 * atr)
            tp = entry + (rr_ratio * 1.5 * atr)
            active_trade = {
                "direction": "BUY",
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr,
                "be_moved": False
            }
            daily_trades_count += 1
            continue

        # SELL Signal
        is_bear_trend = (ema20 < ema50) and (c_p < ema100)
        pulled_back_resist = any(h >= (ema20 - 0.75 * atr) for h in recent_highs)
        rsi_bear = 38.0 <= rsi <= 60.0
        trigger_bear = (c_p < o_p) and (c_p < ema20)

        if is_bear_trend and pulled_back_resist and rsi_bear and trigger_bear:
            entry = opens[i]
            sl = entry + (1.5 * atr)
            tp = entry - (rr_ratio * 1.5 * atr)
            active_trade = {
                "direction": "SELL",
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr,
                "be_moved": False
            }
            daily_trades_count += 1
            continue

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {"trades": 0, "win_rate": 0, "net_pnl": 0, "profit_factor": 0, "max_dd": 0}

    wins = len(tdf[tdf["result"] == "WIN"])
    losses = len(tdf[tdf["result"] == "LOSS"])
    bes = len(tdf[tdf["result"] == "BE"])
    total = len(tdf)
    win_rate = (wins / total) * 100.0
    net_pnl = tdf["pnl"].sum()

    gross_profit = wins * (risk_per_trade * rr_ratio)
    gross_loss = losses * risk_per_trade
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    tdf["cum_pnl"] = tdf["pnl"].cumsum()
    tdf["peak"] = tdf["cum_pnl"].cummax()
    max_dd = (tdf["peak"] - tdf["cum_pnl"]).max()

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "bes": bes,
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_pnl, 2),
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 2)
    }

def main():
    print("Loading 2-Year GBPUSD data from MetaTrader 5...")
    df = fetch_gbp_data()
    print(f"Loaded {len(df):,} candles.")
    print("=" * 80)
    print(f"{'VARIANT / OPTIMIZATION':<35} | {'TRADES':<6} | {'WIN %':<6} | {'NET PNL':<10} | {'PF':<5} | {'MAX DD'}")
    print("=" * 80)

    # 1. Baseline
    r_base = simulate_strategy(df)
    print(f"{'1. Baseline (Current Strategy)':<35} | {r_base['trades']:<6} | {r_base['win_rate']:<6}% | ${r_base['net_pnl']:<+9.2f} | {r_base['profit_factor']:<5} | ${r_base['max_dd']}")

    # 2. Strong Body Candle Filter
    r_body = simulate_strategy(df, filter_strong_body=True)
    print(f"{'2. Strong Candle Body (>=40% range)':<35} | {r_body['trades']:<6} | {r_body['win_rate']:<6}% | ${r_body['net_pnl']:<+9.2f} | {r_body['profit_factor']:<5} | ${r_body['max_dd']}")

    # 3. Breakeven at +1R
    r_be = simulate_strategy(df, move_to_breakeven=True)
    print(f"{'3. Move Stop to Breakeven @ +1R':<35} | {r_be['trades']:<6} | {r_be['win_rate']:<6}% | ${r_be['net_pnl']:<+9.2f} | {r_be['profit_factor']:<5} | ${r_be['max_dd']}")

    # 4. ADX Filter > 20
    r_adx = simulate_strategy(df, filter_adx=True, adx_min=20.0)
    print(f"{'4. ADX Trend Filter (> 20)':<35} | {r_adx['trades']:<6} | {r_adx['win_rate']:<6}% | ${r_adx['net_pnl']:<+9.2f} | {r_adx['profit_factor']:<5} | ${r_adx['max_dd']}")

    # 5. Combined: Strong Body + Breakeven @ +1R
    r_combo1 = simulate_strategy(df, filter_strong_body=True, move_to_breakeven=True)
    print(f"{'5. Strong Body + Breakeven @ +1R':<35} | {r_combo1['trades']:<6} | {r_combo1['win_rate']:<6}% | ${r_combo1['net_pnl']:<+9.2f} | {r_combo1['profit_factor']:<5} | ${r_combo1['max_dd']}")

    # 6. Combined: Strong Body + R:R 1:1.75
    r_rr = simulate_strategy(df, filter_strong_body=True, rr_ratio=1.75)
    print(f"{'6. Strong Body + 1:1.75 R:R Ratio':<35} | {r_rr['trades']:<6} | {r_rr['win_rate']:<6}% | ${r_rr['net_pnl']:<+9.2f} | {r_rr['profit_factor']:<5} | ${r_rr['max_dd']}")

    # 7. Combined: Strong Body + ADX > 20 + 1:2 R:R
    r_combo2 = simulate_strategy(df, filter_strong_body=True, filter_adx=True, adx_min=20.0)
    print(f"{'7. Strong Body + ADX > 20':<35} | {r_combo2['trades']:<6} | {r_combo2['win_rate']:<6}% | ${r_combo2['net_pnl']:<+9.2f} | {r_combo2['profit_factor']:<5} | ${r_combo2['max_dd']}")
    print("=" * 80)

if __name__ == "__main__":
    main()

