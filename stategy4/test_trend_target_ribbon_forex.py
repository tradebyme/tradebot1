import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import MetaTrader5 as mt5


def fast_alma(series: np.ndarray, length: int = 34, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """Arnaud Legoux Moving Average (ALMA) via fast 1D convolution."""
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()

    n = len(series)
    res = np.full(n, np.nan)
    conv = np.convolve(series, weights[::-1], mode="valid")
    res[length - 1:] = conv
    return res


def backtest_trend_target_ribbon(
    symbol: str = "GBPUSD",
    bars_count: int = 50000,
    tp_mode: str = "single_2r",  # "single_2r", "single_1.5r", "scale_out", "stop_flip_only"
    be_after_t1: bool = False,
    use_session_filter: bool = False,
    use_ema200: bool = False,
    capital: float = 2000.0,
    risk_per_trade_usd: float = 25.0,  # 1.25% on $2,000 capital
    slippage_pips: float = 0.5
) -> Dict[str, Any]:
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_count)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return {"error": "No rates"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    n = len(df)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].to_numpy()

    pip_unit = 0.01 if "JPY" in symbol else 0.0001
    slip = slippage_pips * pip_unit

    # 1. Indicators
    alma = fast_alma(closes, length=34, offset=0.85, sigma=6.0)
    dev = df["close"].rolling(window=34, min_periods=1).std().values

    # ATR 14
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().values

    # 200 EMA
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values

    # Slope Score: (alma - alma[slopeLen]) / atr
    slope_len = 3
    shift_alma = np.roll(alma, slope_len)
    shift_alma[:slope_len] = np.nan
    slope_score = np.where((atr > 0) & (~np.isnan(alma)) & (~np.isnan(shift_alma)),
                           (alma - shift_alma) / atr, 0.0)

    dev_mult = 0.65
    upper_confirm = alma + dev * dev_mult
    lower_confirm = alma - dev * dev_mult

    slope_min = 0.08
    bull_setup = (slope_score > slope_min) & (closes > upper_confirm)
    bear_setup = (slope_score < -slope_min) & (closes < lower_confirm)

    # State machine
    trend = 0
    bull_flip = np.zeros(n, dtype=bool)
    bear_flip = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if trend != 1 and bull_setup[i]:
            trend = 1
            bull_flip[i] = True
        elif trend != -1 and bear_setup[i]:
            trend = -1
            bear_flip[i] = True

    # Lowest low / highest high over stopLookback=12
    stop_lookback = 12
    struct_low = df["low"].rolling(window=stop_lookback).min().shift(1).values
    struct_high = df["high"].rolling(window=stop_lookback).max().shift(1).values

    min_stop_atr = 0.75
    max_stop_atr = 3.0

    hours = pd.to_datetime(times).hour
    if use_session_filter:
        session_mask = (hours >= 7) & (hours < 17)
    else:
        session_mask = np.ones(n, dtype=bool)

    trades = []
    active_trade = None
    current_equity = capital
    peak_equity = capital
    max_drawdown = 0.0

    for i in range(50, n):
        bar_time = times[i]
        bar_close = closes[i]
        bar_high = highs[i]
        bar_low = lows[i]
        curr_atr = atr[i]
        is_bull = bull_flip[i]
        is_bear = bear_flip[i]
        in_session = session_mask[i]
        curr_ema = ema200[i]

        # Manage active trade
        if active_trade is not None:
            t_dir = active_trade["direction"]
            entry_p = active_trade["entry_price"]
            stop_p = active_trade["stop_lvl"]
            risk_dist = active_trade["risk_dist"]
            be_active = active_trade["be_active"]
            units = active_trade["units"]

            # Breakeven check after 1R
            if be_after_t1 and not be_active:
                if t_dir == "BUY" and bar_high >= (entry_p + risk_dist):
                    active_trade["be_active"] = True
                    active_trade["stop_lvl"] = entry_p
                    stop_p = entry_p
                elif t_dir == "SELL" and bar_low <= (entry_p - risk_dist):
                    active_trade["be_active"] = True
                    active_trade["stop_lvl"] = entry_p
                    stop_p = entry_p

            if t_dir == "BUY":
                # Check Stop Loss
                if bar_low <= stop_p:
                    exit_p = stop_p - slip
                    pnl = (exit_p - entry_p) * units
                    active_trade.update({"exit_price": exit_p, "exit_time": bar_time, "pnl": round(pnl, 2), "result": "STOP_LOSS"})
                    current_equity += pnl
                    trades.append(active_trade)
                    active_trade = None
                # Check Take Profit
                elif tp_mode in ("single_1.5r", "single_2r"):
                    target_mult = 1.5 if tp_mode == "single_1.5r" else 2.0
                    tp_price = entry_p + risk_dist * target_mult
                    if bar_high >= tp_price:
                        exit_p = tp_price - slip
                        pnl = (exit_p - entry_p) * units
                        active_trade.update({"exit_price": exit_p, "exit_time": bar_time, "pnl": round(pnl, 2), "result": "TAKE_PROFIT"})
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None

            elif t_dir == "SELL":
                # Check Stop Loss
                if bar_high >= stop_p:
                    exit_p = stop_p + slip
                    pnl = (entry_p - exit_p) * units
                    active_trade.update({"exit_price": exit_p, "exit_time": bar_time, "pnl": round(pnl, 2), "result": "STOP_LOSS"})
                    current_equity += pnl
                    trades.append(active_trade)
                    active_trade = None
                # Check Take Profit
                elif tp_mode in ("single_1.5r", "single_2r"):
                    target_mult = 1.5 if tp_mode == "single_1.5r" else 2.0
                    tp_price = entry_p - risk_dist * target_mult
                    if bar_low <= tp_price:
                        exit_p = tp_price + slip
                        pnl = (entry_p - exit_p) * units
                        active_trade.update({"exit_price": exit_p, "exit_time": bar_time, "pnl": round(pnl, 2), "result": "TAKE_PROFIT"})
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None

        # Check Trend Flip / New Entry
        if (is_bull or is_bear) and curr_atr > 0.00005:
            # Reversal: close active trade if opposing flip
            if active_trade is not None:
                if (active_trade["direction"] == "BUY" and is_bear) or (active_trade["direction"] == "SELL" and is_bull):
                    exit_p = (bar_close - slip) if active_trade["direction"] == "BUY" else (bar_close + slip)
                    pnl = (exit_p - active_trade["entry_price"]) * active_trade["units"] if active_trade["direction"] == "BUY" else (active_trade["entry_price"] - exit_p) * active_trade["units"]
                    active_trade.update({"exit_price": exit_p, "exit_time": bar_time, "pnl": round(pnl, 2), "result": "TREND_FLIP"})
                    current_equity += pnl
                    trades.append(active_trade)
                    active_trade = None

            # New Entry
            ema_ok_long = (not use_ema200) or (bar_close > curr_ema)
            ema_ok_short = (not use_ema200) or (bar_close < curr_ema)

            if is_bull and in_session and ema_ok_long and active_trade is None:
                entry_p = bar_close + slip
                raw_risk = bar_close - struct_low[i]
                min_risk = curr_atr * min_stop_atr
                max_risk = curr_atr * max_stop_atr
                risk_dist = min(max(raw_risk if not np.isnan(raw_risk) else min_risk, min_risk), max_risk)
                stop_lvl = entry_p - risk_dist
                units = max(1000.0, float(int(risk_per_trade_usd / risk_dist)))

                active_trade = {
                    "direction": "BUY",
                    "entry_time": bar_time,
                    "entry_price": entry_p,
                    "stop_lvl": stop_lvl,
                    "risk_dist": risk_dist,
                    "units": units,
                    "be_active": False
                }

            elif is_bear and in_session and ema_ok_short and active_trade is None:
                entry_p = bar_close - slip
                raw_risk = struct_high[i] - bar_close
                min_risk = curr_atr * min_stop_atr
                max_risk = curr_atr * max_stop_atr
                risk_dist = min(max(raw_risk if not np.isnan(raw_risk) else min_risk, min_risk), max_risk)
                stop_lvl = entry_p + risk_dist
                units = max(1000.0, float(int(risk_per_trade_usd / risk_dist)))

                active_trade = {
                    "direction": "SELL",
                    "entry_time": bar_time,
                    "entry_price": entry_p,
                    "stop_lvl": stop_lvl,
                    "risk_dist": risk_dist,
                    "units": units,
                    "be_active": False
                }

        # Track drawdown
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = peak_equity - current_equity
        if dd > max_drawdown:
            max_drawdown = dd

    # Close open trade at end
    if active_trade is not None:
        last_p = closes[-1]
        pnl = (last_p - active_trade["entry_price"]) * active_trade["units"] if active_trade["direction"] == "BUY" else (active_trade["entry_price"] - last_p) * active_trade["units"]
        active_trade.update({"exit_price": last_p, "exit_time": times[-1], "pnl": round(pnl, 2), "result": "END_OF_DATA"})
        current_equity += pnl
        trades.append(active_trade)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if trades_df.empty:
        return {"total_trades": 0, "net_profit": 0.0}

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = (gp / gl) if gl > 0 else 999.0
    net_pnl = current_equity - capital
    return_pct = (net_pnl / capital) * 100.0

    return {
        "symbol": symbol,
        "bars": n,
        "start": str(df["time"].iloc[0])[:10],
        "end": str(df["time"].iloc[-1])[:10],
        "total_trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(return_pct, 1),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown / capital * 100.0, 1),
        "trades_df": trades_df
    }


def run_forex_ribbon_study():
    symbols = ["GBPUSD", "EURUSD", "USDJPY"]
    modes = [
        ("Stop & Flip Only (24/5)", "stop_flip_only", False, False, False),
        ("Single Target 1:2 R:R (24/5)", "single_2r", False, False, False),
        ("Single Target 1:1.5 R:R (24/5)", "single_1.5r", False, False, False),
        ("London/NY Session (07-17 UTC) + 1:2 R:R", "single_2r", False, True, False),
        ("London/NY Session + 200 EMA Filter + 1:2 R:R", "single_2r", False, True, True),
        ("London/NY Session + Breakeven After 1R + 1:2 R:R", "single_2r", True, True, False)
    ]

    print("\n" + "=" * 80)
    print(" 🚀 TREND TARGET RIBBON STRATEGY [BOSWaves] — MULTI-YEAR FOREX TEST (MT5)")
    print("=" * 80)

    for sym in symbols:
        print(f"\n==================== [ {sym} ] ====================")
        for label, tp_mode, be, session, ema in modes:
            res = backtest_trend_target_ribbon(
                symbol=sym,
                bars_count=50000,
                tp_mode=tp_mode,
                be_after_t1=be,
                use_session_filter=session,
                use_ema200=ema
            )
            print(f" • {label}:")
            print(f"     Trades: {res['total_trades']:,} | Win Rate: {res['win_rate']}% ({res['wins']}W / {res['losses']}L)")
            print(f"     Net PnL: ${res['net_pnl']:+,.2f} ({res['return_pct']:+.1f}%) | Profit Factor: {res['profit_factor']:.2f}")
            print(f"     Max Drawdown: -${res['max_drawdown']:,.2f} ({res['max_drawdown_pct']:.1f}%)")
            print()


if __name__ == "__main__":
    run_forex_ribbon_study()

