import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def fetch_mt5_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch M15 candles with aligned H1 trend filter from MetaTrader 5."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 Init failed: {mt5.last_error()}")

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)

    rates_m15 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
    mt5.shutdown()

    if rates_m15 is None or len(rates_m15) == 0:
        raise ValueError(f"No M15 rates for {symbol}")

    df_m15 = pd.DataFrame(rates_m15)
    df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')

    # H1 indicators for trend alignment
    df_h1 = pd.DataFrame(rates_h1)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1['h1_ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
    # Shift by 1 to use only completed H1 candles (zero lookahead)
    df_h1['h1_bullish'] = (df_h1['close'].shift(1) > df_h1['h1_ema50'].shift(1)).astype(float)
    df_h1['h1_bearish'] = (df_h1['close'].shift(1) < df_h1['h1_ema50'].shift(1)).astype(float)

    # Merge H1 trend into M15
    df_merged = pd.merge_asof(
        df_m15.sort_values('time'),
        df_h1[['time', 'h1_bullish', 'h1_bearish']].sort_values('time'),
        on='time',
        direction='backward'
    )

    # M15 Indicators
    df = df_merged.copy()
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
    df["adx_14"] = compute_adx(df, 14)

    return df

def simulate_strategy(df: pd.DataFrame, is_upgraded: bool = False, capital: float = 2000.0) -> dict:
    """
    Simulate bar-by-bar execution.
    - is_upgraded=False: Baseline Strategy 1 (M15 only, 1.25% risk, no breakeven, loose pullback)
    - is_upgraded=True:  Upgraded Strategy 1 (H1 trend filter, strict pullback, ADX > 20, Breakeven at +1 ATR, 0.75% risk)
    """
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
    h1_bull = df["h1_bullish"].values
    h1_bear = df["h1_bearish"].values

    risk_usd = 15.0 if is_upgraded else 25.0  # 0.75% vs 1.25%
    rr_ratio = 2.0  # 1:2 R:R

    trades = []
    active_trade = None
    last_trade_day = None
    daily_trades_count = 0
    daily_pnl = 0.0
    circuit_breaker = False

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
            circuit_breaker = False

        # --- 1. Manage Active Position ---
        if active_trade is not None:
            curr_low = lows[i]
            curr_high = highs[i]
            entry = active_trade["entry_price"]
            direction = active_trade["direction"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            atr = active_trade["atr"]

            # Dynamic Breakeven Check (Upgraded model only)
            if is_upgraded and not active_trade.get("be_moved", False):
                if direction == "BUY" and curr_high >= entry + (1.0 * atr):
                    active_trade["sl"] = entry + (0.0001)  # Move SL to Entry + 1 pip
                    active_trade["be_moved"] = True
                elif direction == "SELL" and curr_low <= entry - (1.0 * atr):
                    active_trade["sl"] = entry - (0.0001)  # Move SL to Entry + 1 pip
                    active_trade["be_moved"] = True
                sl = active_trade["sl"]

            # Check SL & TP fills
            if direction == "BUY":
                if curr_low <= sl:
                    is_be = active_trade.get("be_moved", False) and abs(sl - entry) < 0.0003
                    pnl = 0.0 if is_be else -risk_usd
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": "BUY",
                        "pnl": pnl,
                        "result": "BE" if is_be else "LOSS"
                    })
                    daily_pnl += pnl
                    active_trade = None
                elif curr_high >= tp:
                    pnl = risk_usd * rr_ratio
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": "BUY",
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    daily_pnl += pnl
                    active_trade = None

            elif direction == "SELL":
                if curr_high >= sl:
                    is_be = active_trade.get("be_moved", False) and abs(sl - entry) < 0.0003
                    pnl = 0.0 if is_be else -risk_usd
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": "SELL",
                        "pnl": pnl,
                        "result": "BE" if is_be else "LOSS"
                    })
                    daily_pnl += pnl
                    active_trade = None
                elif curr_low <= tp:
                    pnl = risk_usd * rr_ratio
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": "SELL",
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    daily_pnl += pnl
                    active_trade = None

            # Circuit breaker check: daily loss >= 3.5% ($70)
            if daily_pnl <= -70.0:
                circuit_breaker = True

        if active_trade is not None:
            continue

        # --- 2. Session & Day Filters ---
        if not (7 <= bar_hour < 17):
            continue
        if weekday in (5, 6) or (weekday == 4 and bar_hour >= 20):
            continue
        if circuit_breaker or daily_trades_count >= 3:
            continue

        # --- 3. Signal Generation on Completed Bar (i - 1) ---
        prev = i - 1
        atr = atr_vals[prev]
        rsi = rsi_vals[prev]
        ema20 = ema20_vals[prev]
        ema50 = ema50_vals[prev]
        ema100 = ema100_vals[prev]
        c_p = closes[prev]
        o_p = opens[prev]
        adx = adx_vals[prev]

        if np.isnan(atr) or atr <= 0.0001:
            continue

        recent_lows = lows[i-3:i]
        recent_highs = highs[i-3:i]

        if not is_upgraded:
            # === BASELINE STRATEGY 1 ===
            is_bull = (ema20 > ema50) and (c_p > ema100)
            pb_bull = any(l <= (ema20 + 0.75 * atr) for l in recent_lows)
            rsi_bull = 40.0 <= rsi <= 62.0
            trig_bull = (c_p > o_p) and (c_p > ema20)

            if is_bull and pb_bull and rsi_bull and trig_bull:
                entry = opens[i]
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "BUY",
                    "entry_price": entry,
                    "sl": entry - (1.5 * atr),
                    "tp": entry + (3.0 * atr),
                    "atr": atr,
                    "be_moved": False
                }
                daily_trades_count += 1
                continue

            is_bear = (ema20 < ema50) and (c_p < ema100)
            pb_bear = any(h >= (ema20 - 0.75 * atr) for h in recent_highs)
            rsi_bear = 38.0 <= rsi <= 60.0
            trig_bear = (c_p < o_p) and (c_p < ema20)

            if is_bear and pb_bear and rsi_bear and trig_bear:
                entry = opens[i]
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "SELL",
                    "entry_price": entry,
                    "sl": entry + (1.5 * atr),
                    "tp": entry - (3.0 * atr),
                    "atr": atr,
                    "be_moved": False
                }
                daily_trades_count += 1
                continue

        else:
            # === UPGRADED STRATEGY 1 ===
            # Filter 1: H1 Trend Alignment
            # Filter 2: ADX > 20
            # Filter 3: Strict Pullback (Low must touch or cross EMA 20)
            if adx < 20.0:
                continue

            # Bullish
            h1_ok_bull = (h1_bull[prev] == 1.0)
            is_bull = h1_ok_bull and (ema20 > ema50) and (c_p > ema100)
            strict_pb_bull = any(l <= ema20 for l in recent_lows)
            rsi_bull = 40.0 <= rsi <= 62.0
            trig_bull = (c_p > o_p) and (c_p > ema20)

            if is_bull and strict_pb_bull and rsi_bull and trig_bull:
                entry = opens[i]
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "BUY",
                    "entry_price": entry,
                    "sl": entry - (1.5 * atr),
                    "tp": entry + (3.0 * atr),
                    "atr": atr,
                    "be_moved": False
                }
                daily_trades_count += 1
                continue

            # Bearish
            h1_ok_bear = (h1_bear[prev] == 1.0)
            is_bear = h1_ok_bear and (ema20 < ema50) and (c_p < ema100)
            strict_pb_bear = any(h >= ema20 for h in recent_highs)
            rsi_bear = 38.0 <= rsi <= 60.0
            trig_bear = (c_p < o_p) and (c_p < ema20)

            if is_bear and strict_pb_bear and rsi_bear and trig_bear:
                entry = opens[i]
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "SELL",
                    "entry_price": entry,
                    "sl": entry + (1.5 * atr),
                    "tp": entry - (3.0 * atr),
                    "atr": atr,
                    "be_moved": False
                }
                daily_trades_count += 1
                continue

    # Analysis
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return {"total_trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "profit_factor": 0.0, "wins": 0, "losses": 0, "be_count": 0}

    total = len(trades_df)
    wins = len(trades_df[trades_df["result"] == "WIN"])
    losses = len(trades_df[trades_df["result"] == "LOSS"])
    be_count = len(trades_df[trades_df["result"] == "BE"])
    win_rate = (wins / total) * 100.0 if total > 0 else 0.0
    net_pnl = float(trades_df["pnl"].sum())

    gross_win = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
    gross_loss = abs(float(trades_df[trades_df["pnl"] < 0]["pnl"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    trades_df["cum"] = trades_df["pnl"].cumsum()
    trades_df["peak"] = trades_df["cum"].cummax()
    trades_df["dd"] = trades_df["peak"] - trades_df["cum"]
    max_dd = float(trades_df["dd"].max())
    max_dd_pct = (max_dd / capital) * 100.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "be_count": be_count,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "final_equity": capital + net_pnl
    }

def run_study():
    horizons = [
        ("Short-Term (90 Days / 3 Months)", 90),
        ("Medium-Term (1 Year / 365 Days)", 365),
        ("Long-Term (2 Years / 730 Days)", 730),
        ("Multi-Year (4 Years / 1460 Days)", 1460)
    ]
    symbols = ["GBPUSD", "EURUSD"]

    print("=" * 84)
    print("      COMPREHENSIVE BACKTEST REPORT: BASELINE vs UPGRADED STRATEGY 1      ")
    print("      Engine: MetaTrader 5 High-Precision M15 & H1 Real Forex Feeds        ")
    print("      Prop Firm Challenge Constraints: $2,000 Capital | 8% Max Drawdown   ")
    print("=" * 84)

    all_results = []

    for sym in symbols:
        print(f"\n========================== ASSET: {sym} ==========================")
        for label, days in horizons:
            print(f"\n>>> Running Study: {sym} | {label} (Fetching MT5 data...)")
            df = fetch_mt5_data(sym, days)
            m15_count = len(df)
            start_date = df['time'].iloc[0].strftime('%Y-%m-%d')
            end_date = df['time'].iloc[-1].strftime('%Y-%m-%d')

            res_base = simulate_strategy(df, is_upgraded=False)
            res_upg = simulate_strategy(df, is_upgraded=True)

            all_results.append({
                "symbol": sym,
                "horizon": label,
                "bars": m15_count,
                "start": start_date,
                "end": end_date,
                "base": res_base,
                "upg": res_upg
            })

            print(f"    Date Range: {start_date} to {end_date} ({m15_count:,} M15 bars)")
            print(f"    {'Metric':<25} | {'Baseline (Original)':<22} | {'Upgraded (Enhanced)':<22}")
            print(f"    {'-'*25}-|-{'-'*22}-|-{'-'*22}")
            b_w = res_base['wins']
            b_l = res_base['losses']
            b_be = res_base['be_count']
            u_w = res_upg['wins']
            u_l = res_upg['losses']
            u_be = res_upg['be_count']

            b_wlbl = f"{b_w}W / {b_l}L / {b_be}BE"
            u_wlbl = f"{u_w}W / {u_l}L / {u_be}BE"
            b_wr = f"{res_base['win_rate']:.1f}%"
            u_wr = f"{res_upg['win_rate']:.1f}%"
            b_pnl = f"${res_base['net_pnl']:,.2f}"
            u_pnl = f"${res_upg['net_pnl']:,.2f}"
            b_pf = f"{res_base['profit_factor']:.2f}"
            u_pf = f"{res_upg['profit_factor']:.2f}"
            b_dd = f"${res_base['max_drawdown']:,.2f} ({res_base['max_drawdown_pct']:.1f}%)"
            u_dd = f"${res_upg['max_drawdown']:,.2f} ({res_upg['max_drawdown_pct']:.1f}%)"

            base_passed = "PASSED" if res_base['max_drawdown_pct'] < 8.0 else "FAILED (Hit 8% DD)"
            upg_passed = "PASSED" if res_upg['max_drawdown_pct'] < 8.0 else "FAILED (Hit 8% DD)"

            print(f"    {'Total Trades':<25} | {res_base['total_trades']:<22} | {res_upg['total_trades']:<22}")
            print(f"    {'Wins / Losses / BE':<25} | {b_wlbl:<22} | {u_wlbl:<22}")
            print(f"    {'Win Rate (%)':<25} | {b_wr:<22} | {u_wr:<22}")
            print(f"    {'Net Profit ($)':<25} | {b_pnl:<22} | {u_pnl:<22}")
            print(f"    {'Profit Factor':<25} | {b_pf:<22} | {u_pf:<22}")
            print(f"    {'Max Drawdown ($)':<25} | {b_dd:<22} | {u_dd:<22}")
            print(f"    {'8% Prop DD Status':<25} | {base_passed:<22} | {upg_passed:<22}")

if __name__ == "__main__":
    run_study()
