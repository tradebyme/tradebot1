import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import MetaTrader5 as mt5


def fetch_mt5_candles(symbol: str, count: int = 100000) -> pd.DataFrame:
    """Fetch up to 100,000 completed M15 candles from MetaTrader 5."""
    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader 5 init failed: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"No rates returned for {symbol}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def backtest_strategy1_forex(
    df: pd.DataFrame,
    symbol: str = "GBPUSD",
    capital: float = 2000.0,
    risk_per_trade_usd: float = 25.0,  # 1.25% on $2,000
    rr_ratio: float = 2.0,             # 1:2 R:R (Win = +$50, Loss = -$25)
    max_daily_losses: int = 2,
    daily_circuit_breaker_usd: float = 70.0
) -> Dict[str, Any]:
    n = len(df)

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    opens = df["open"]

    # 1. EMAs (20, 50, 100)
    ema20 = closes.ewm(span=20, adjust=False).mean().values
    ema50 = closes.ewm(span=50, adjust=False).mean().values
    ema100 = closes.ewm(span=100, adjust=False).mean().values

    # 2. ATR 14 (SMA of True Range, exactly matching Strategy 1 python bot)
    prev_close = closes.shift(1)
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().values

    # 3. RSI 14
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).values

    # 4. Pullback Lookback (3 candles)
    recent_min_low = lows.rolling(window=3, min_periods=1).min().values
    recent_max_high = highs.rolling(window=3, min_periods=1).max().values

    times_arr = df["time"].to_numpy()
    highs_arr = highs.values
    lows_arr = lows.values
    closes_arr = closes.values
    opens_arr = opens.values

    # Session filter: London 07:00 UTC to NY 17:00 UTC, weekdays only
    dts = pd.to_datetime(times_arr)
    hours = dts.hour
    weekdays = dts.weekday  # 0=Monday ... 4=Friday, 5=Saturday, 6=Sunday
    is_weekend = (weekdays == 5) | ((weekdays == 6) & (hours < 21)) | ((weekdays == 4) & (hours >= 21))
    in_session = (hours >= 7) & (hours < 17) & (~is_weekend)

    dates = dts.date
    unique_dates = sorted(list(set(dates)))

    trades = []
    active_trade = None
    last_trade_day = None
    daily_losses = 0
    daily_pnl = 0.0

    current_equity = capital
    peak_equity = capital
    max_drawdown = 0.0

    pip_unit = 0.01 if "JPY" in symbol else 0.0001

    for i in range(100, n):
        bar_time = times_arr[i]
        bar_date = dates[i]
        bar_close = closes_arr[i]
        bar_high = highs_arr[i]
        bar_low = lows_arr[i]
        bar_open = opens_arr[i]
        curr_atr = atr[i]
        curr_rsi = rsi[i]
        e20 = ema20[i]
        e50 = ema50[i]
        e100 = ema100[i]
        sess_ok = in_session[i]

        # Daily Rollover check (00:00 UTC)
        if bar_date != last_trade_day:
            last_trade_day = bar_date
            daily_losses = 0
            daily_pnl = 0.0

        # 1. Manage Active Trade
        if active_trade is not None:
            t_dir = active_trade["direction"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]

            if t_dir == "BUY":
                if bar_low <= sl:
                    pnl = -risk_per_trade_usd
                    active_trade.update({
                        "exit_price": sl,
                        "exit_time": bar_time,
                        "pnl": pnl,
                        "result": "LOSS"
                    })
                    current_equity += pnl
                    daily_pnl += pnl
                    daily_losses += 1
                    trades.append(active_trade)
                    active_trade = None
                elif bar_high >= tp:
                    pnl = risk_per_trade_usd * rr_ratio
                    active_trade.update({
                        "exit_price": tp,
                        "exit_time": bar_time,
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    current_equity += pnl
                    daily_pnl += pnl
                    trades.append(active_trade)
                    active_trade = None

            elif t_dir == "SELL":
                if bar_high >= sl:
                    pnl = -risk_per_trade_usd
                    active_trade.update({
                        "exit_price": sl,
                        "exit_time": bar_time,
                        "pnl": pnl,
                        "result": "LOSS"
                    })
                    current_equity += pnl
                    daily_pnl += pnl
                    daily_losses += 1
                    trades.append(active_trade)
                    active_trade = None
                elif bar_low <= tp:
                    pnl = risk_per_trade_usd * rr_ratio
                    active_trade.update({
                        "exit_price": tp,
                        "exit_time": bar_time,
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    current_equity += pnl
                    daily_pnl += pnl
                    trades.append(active_trade)
                    active_trade = None

        # Track Peak Equity and Drawdown
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = peak_equity - current_equity
        if dd > max_drawdown:
            max_drawdown = dd

        # 2. Check Entries (if no active trade and circuit breaker not triggered)
        circuit_tripped = (daily_losses >= max_daily_losses) or (daily_pnl <= -daily_circuit_breaker_usd)

        if active_trade is None and not circuit_tripped and sess_ok and curr_atr > 0.00005:
            # Bullish Setup
            # Trend: 20 EMA > 50 EMA and Close > 100 EMA
            # Pullback: Low in last 3 candles touched within 0.75 ATR of EMA 20
            # RSI: 40.0 <= rsi <= 62.0
            # Trigger: Green candle (close > open) and close > ema20
            is_bull = (
                (e20 > e50) and
                (bar_close > e100) and
                (recent_min_low[i] <= (e20 + 0.75 * curr_atr)) and
                (40.0 <= curr_rsi <= 62.0) and
                (bar_close > bar_open) and
                (bar_close > e20)
            )

            # Bearish Setup
            # Trend: 20 EMA < 50 EMA and Close < 100 EMA
            # Pullback: High in last 3 candles touched within 0.75 ATR of EMA 20
            # RSI: 38.0 <= rsi <= 60.0
            # Trigger: Red candle (close < open) and close < ema20
            is_bear = (
                (e20 < e50) and
                (bar_close < e100) and
                (recent_max_high[i] >= (e20 - 0.75 * curr_atr)) and
                (38.0 <= curr_rsi <= 60.0) and
                (bar_close < bar_open) and
                (bar_close < e20)
            )

            sl_dist = 1.5 * curr_atr
            tp_dist = 3.0 * curr_atr

            if is_bull:
                active_trade = {
                    "symbol": symbol,
                    "direction": "BUY",
                    "entry_time": bar_time,
                    "entry_price": bar_close,
                    "sl": round(bar_close - sl_dist, 5),
                    "tp": round(bar_close + tp_dist, 5),
                    "atr": curr_atr,
                    "date": bar_date
                }
            elif is_bear:
                active_trade = {
                    "symbol": symbol,
                    "direction": "SELL",
                    "entry_time": bar_time,
                    "entry_price": bar_close,
                    "sl": round(bar_close + sl_dist, 5),
                    "tp": round(bar_close - tp_dist, 5),
                    "atr": curr_atr,
                    "date": bar_date
                }

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return _compute_analytics(df, trades_df, capital, current_equity, max_drawdown, symbol, rr_ratio)


def _compute_analytics(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    capital: float,
    ending_equity: float,
    max_drawdown: float,
    symbol: str,
    rr_ratio: float
) -> Dict[str, Any]:
    n_bars = len(df)
    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]
    days_covered = (end_time - start_time).days
    years_covered = round(days_covered / 365.25, 2)

    if trades_df.empty:
        return {"symbol": symbol, "total_trades": 0, "net_profit": 0.0}

    total_trades = len(trades_df)
    wins = trades_df[trades_df["result"] == "WIN"]
    losses = trades_df[trades_df["result"] == "LOSS"]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = win_count * (25.0 * rr_ratio)
    gross_loss = loss_count * 25.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    net_profit = gross_profit - gross_loss
    return_pct = (net_profit / capital) * 100.0
    max_dd_pct = (max_drawdown / capital) * 100.0

    # Consecutive Streaks
    pnl_signs = (trades_df["result"] == "WIN").astype(int)
    max_consec_wins = 0
    max_consec_losses = 0
    curr_win = 0
    curr_loss = 0
    for s in pnl_signs:
        if s == 1:
            curr_win += 1
            curr_loss = 0
            max_consec_wins = max(max_consec_wins, curr_win)
        else:
            curr_loss += 1
            curr_win = 0
            max_consec_losses = max(max_consec_losses, curr_loss)

    # Yearly Breakdown
    trades_df["year"] = pd.to_datetime(trades_df["entry_time"]).dt.year
    yearly_stats = {}
    for yr, group in trades_df.groupby("year"):
        y_w = len(group[group["result"] == "WIN"])
        y_l = len(group[group["result"] == "LOSS"])
        y_gp = y_w * (25.0 * rr_ratio)
        y_gl = y_l * 25.0
        y_pf = (y_gp / y_gl) if y_gl > 0 else 999.0
        yearly_stats[yr] = {
            "trades": len(group),
            "wins": y_w,
            "losses": y_l,
            "win_rate": round(y_w / len(group) * 100.0, 1),
            "net_profit": round(y_gp - y_gl, 2),
            "profit_factor": round(y_pf, 2)
        }

    # Prop Firm Challenge Simulation across rolling windows
    trades_df["date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    unique_days = sorted(list(trades_df["date"].unique()))

    challenge_results = {}
    for window_days in [3, 5, 10]:
        pass_count = 0
        fail_daily_count = 0
        fail_dd_count = 0
        neutral_count = 0
        total_windows = max(0, len(unique_days) - window_days + 1)

        for d_idx in range(total_windows):
            w_start = unique_days[d_idx]
            w_end = unique_days[d_idx + window_days - 1]
            w_trades = trades_df[(trades_df["date"] >= w_start) & (trades_df["date"] <= w_end)]

            if w_trades.empty:
                neutral_count += 1
                continue

            w_pnl = w_trades["pnl"].sum()
            daily_pnls = w_trades.groupby("date")["pnl"].sum()
            worst_day = daily_pnls.min() if not daily_pnls.empty else 0.0

            # 5% daily limit = -$100 on $2,000 capital
            if worst_day <= -100.0:
                fail_daily_count += 1
            # 8% max loss = -$160 on $2,000 capital
            elif w_pnl <= -160.0:
                fail_dd_count += 1
            # +8% target = +$160 on $2,000 capital
            elif w_pnl >= 160.0:
                pass_count += 1
            else:
                neutral_count += 1

        challenge_results[f"{window_days}d"] = {
            "total_windows": total_windows,
            "passed": pass_count,
            "pass_pct": round((pass_count / total_windows * 100.0) if total_windows > 0 else 0.0, 1),
            "failed_daily": fail_daily_count,
            "failed_daily_pct": round((fail_daily_count / total_windows * 100.0) if total_windows > 0 else 0.0, 1),
            "failed_dd": fail_dd_count,
            "failed_dd_pct": round((fail_dd_count / total_windows * 100.0) if total_windows > 0 else 0.0, 1),
            "neutral": neutral_count
        }

    return {
        "symbol": symbol,
        "bars": n_bars,
        "start": start_time.strftime("%Y-%m-%d"),
        "end": end_time.strftime("%Y-%m-%d"),
        "years": years_covered,
        "capital": capital,
        "ending_equity": round(ending_equity, 2),
        "net_profit": round(net_profit, 2),
        "return_pct": round(return_pct, 2),
        "total_trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "yearly_stats": yearly_stats,
        "challenge_results": challenge_results,
        "trades_df": trades_df
    }


def run_full_study():
    symbols = ["GBPUSD", "EURUSD", "USDJPY"]

    print("\n" + "=" * 80)
    print(" 🚀 STRATEGY 1: LONDON/NY MOMENTUM PULLBACK — 4-YEAR FOREX BACKTEST (MT5) ")
    print("=" * 80)

    for sym in symbols:
        print(f"\n==================== [ {sym} ] 4-YEAR BACKTEST ====================")
        df = fetch_mt5_candles(sym, count=100000)
        res = backtest_strategy1_forex(df, symbol=sym)

        print(f" Period: {res['start']} to {res['end']} ({res['years']} Years, {res['bars']:,} M15 Bars)")
        print(f" Capital: ${res['capital']:,.2f} | Risk Per Trade: $25.00 (1.25%) | 1:2 R:R (Win=+$50, Loss=-$25)")
        print("-" * 80)
        print(f" • Total Trades Taken:    {res['total_trades']:,}")
        print(f" • Winning Trades:        {res['wins']:,} (+${res['gross_profit']:,.2f})")
        print(f" • Losing Trades:         {res['losses']:,} (-${res['gross_loss']:,.2f})")
        print(f" • Win Rate:              {res['win_rate']:.2f}% (Needs >33.3% to profit at 1:2 R:R)")
        print(f" • Profit Factor:         {res['profit_factor']:.2f}")
        print(f" • Net Profit:            ${res['net_profit']:+,.2f} ({res['return_pct']:+.1f}% on $2k capital)")
        print(f" • Max Drawdown in 4 Yrs: ${res['max_drawdown']:,.2f} ({res['max_drawdown_pct']:.1f}% of $2k)")
        print(f" • Consecutive Streaks:   Max Wins: {res['max_consec_wins']} | Max Losses: {res['max_consec_losses']}")
        print()
        print(" >> YEAR-BY-YEAR PERFORMANCE BREAKDOWN:")
        for yr, y in res['yearly_stats'].items():
            print(f"    - {yr}: {y['trades']:>3} trades | Win Rate: {y['win_rate']:>4.1f}% | Net: ${y['net_profit']:>+7,.0f} | PF: {y['profit_factor']:.2f}")
        print()
        print(" >> PROP FIRM EVALUATION PASS RATES (Rolling Windows):")
        for wd in ["3d", "5d", "10d"]:
            c = res['challenge_results'][wd]
            print(f"    - [ {wd} Window ]: Passed (+8% / +$160): {c['passed']:>3} ({c['pass_pct']:>4.1f}%) | Daily Breached (-5%): {c['failed_daily']:>2} ({c['failed_daily_pct']:.1f}%) | Max DD Hit (-8%): {c['failed_dd']:>2} ({c['failed_dd_pct']:.1f}%)")
        print("=" * 80)


if __name__ == "__main__":
    run_full_study()

