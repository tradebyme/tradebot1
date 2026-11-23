import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Import AdaptiveSuperSmootherStrategy directly from current directory
from strategy import AdaptiveSuperSmootherStrategy


class InstitutionalMFEBacktester:
    """
    Backtester for Multi-Year Institutional Forex Data from MetaTrader 5.
    Evaluates:
      1. Baseline Fixed SL/TP
      2. Dynamic Breakeven (+0.1R buffer when MFE >= +1.0R)
      3. Dynamic MFE Giveback Exit (Banks profit when unrealized peak retraces by 25-35%)
      4. Smart Trend-Continuation Re-entry
    """

    def __init__(
        self,
        symbol: str = "GBPUSD",
        bars_count: int = 100000,
        initial_capital: float = 10000.0,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        risk_pct: float = 1.0,
        session_filter: bool = True,
        session_start_utc: int = 7,
        session_end_utc: int = 17,
        commission_pct: float = 0.00002,
        slippage_pips: float = 0.5,
    ):
        self.symbol = symbol.replace("/", "").replace("_", "")
        self.bars_count = bars_count
        self.capital = initial_capital
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.risk_pct = risk_pct
        self.session_filter = session_filter
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.commission_pct = commission_pct
        self.slippage_pips = slippage_pips
        self.pip_unit = 0.01 if "JPY" in self.symbol else (0.10 if "XAU" in self.symbol else 0.0001)

    def fetch_mt5_data(self) -> pd.DataFrame:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, self.bars_count)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates returned for {self.symbol}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR 14
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14, min_periods=1).mean()

        # EMAs
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["ema_100"] = close.ewm(span=100, adjust=False).mean()
        df["ema_200"] = close.ewm(span=200, adjust=False).mean()

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # SuperSmoother
        smoother = AdaptiveSuperSmootherStrategy.calculate_adaptive_supersmoother(close)
        df["ss_long"] = smoother["long_signal"]
        df["ss_short"] = smoother["short_signal"]
        df["ss_sq"] = smoother["sq"]
        df["ss_eff"] = smoother["efficiency"]

        return df

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy_mode: str = "strat1",  # "strat1", "strat4", "hybrid"
        use_be: bool = True,
        be_trigger_r: float = 1.0,
        be_lock_r: float = 0.1,
        use_giveback_exit: bool = True,
        giveback_threshold_pct: float = 0.30,  # exit if profit drops 30% from peak
        giveback_activation_r: float = 1.5,   # activate once profit hits +1.5R
        allow_reentry: bool = True,
        max_reentries: int = 1
    ) -> Dict[str, Any]:
        n = len(df)
        times = df["time"].to_numpy()
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df["atr"].values
        ema20s = df["ema_20"].values
        ema50s = df["ema_50"].values
        ema100s = df["ema_100"].values
        ema200s = df["ema_200"].values
        rsis = df["rsi"].values
        ss_longs = df["ss_long"].values
        ss_shorts = df["ss_short"].values
        ss_sqs = df["ss_sq"].values
        ss_effs = df["ss_eff"].values

        hours = pd.to_datetime(times).hour
        if self.session_filter:
            session_mask = (hours >= self.session_start_utc) & (hours < self.session_end_utc)
        else:
            session_mask = np.ones(n, dtype=bool)

        slip = self.slippage_pips * self.pip_unit
        current_equity = self.capital
        peak_equity = self.capital
        max_dd_usd = 0.0

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None
        consecutive_reentries = 0
        last_exit_direction = None
        last_exit_bar = -999

        for i in range(120, n):
            bar_time = times[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            atr = atrs[i]
            in_session = session_mask[i]

            # -------------------------------------------------------------
            # 1. MANAGE ACTIVE TRADE (Intra-bar MFE tracking and Exits)
            # -------------------------------------------------------------
            if active_trade is not None:
                entry_p = active_trade["entry_price"]
                direction = active_trade["direction"]
                units = active_trade["units"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                r_dist = active_trade["r_dist"]
                peak_favorable_price = active_trade["peak_favorable_price"]
                peak_r = active_trade["peak_r"]
                is_be_active = active_trade["is_be_active"]

                # Update High Water Mark (Peak MFE)
                if direction == "BUY":
                    if bar_high > peak_favorable_price:
                        peak_favorable_price = bar_high
                        peak_r = (peak_favorable_price - entry_p) / r_dist
                        active_trade["peak_favorable_price"] = peak_favorable_price
                        active_trade["peak_r"] = peak_r

                    # Check Breakeven Shift
                    if use_be and (not is_be_active) and peak_r >= be_trigger_r:
                        new_sl = entry_p + (be_lock_r * r_dist)
                        if new_sl > sl:
                            sl = new_sl
                            active_trade["sl"] = sl
                            active_trade["is_be_active"] = True

                    # Check MFE Giveback Exit
                    # If peak profit exceeded activation threshold, check if current bar gives back too much
                    giveback_exit_triggered = False
                    giveback_exit_price = bar_close
                    if use_giveback_exit and peak_r >= giveback_activation_r:
                        # Allowable floor: peak gain minus allowed retracement percentage
                        peak_profit_dist = peak_favorable_price - entry_p
                        giveback_floor = peak_favorable_price - (peak_profit_dist * giveback_threshold_pct)
                        if bar_low <= giveback_floor:
                            giveback_exit_triggered = True
                            giveback_exit_price = min(bar_open, giveback_floor)

                    # Order of checks: Stop Loss, Giveback, Take Profit
                    if bar_low <= sl:
                        exit_p = sl - slip
                        raw_pnl = (exit_p - entry_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        res_label = "BREAKEVEN" if is_be_active and net_pnl >= 0 else "STOP_LOSS"
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": res_label, "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

                    elif giveback_exit_triggered:
                        exit_p = giveback_exit_price - slip
                        raw_pnl = (exit_p - entry_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": "MFE_GIVEBACK_EXIT", "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

                    elif tp is not None and bar_high >= tp:
                        exit_p = tp - slip
                        raw_pnl = (exit_p - entry_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": "TAKE_PROFIT", "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

                elif direction == "SELL":
                    if bar_low < peak_favorable_price:
                        peak_favorable_price = bar_low
                        peak_r = (entry_p - peak_favorable_price) / r_dist
                        active_trade["peak_favorable_price"] = peak_favorable_price
                        active_trade["peak_r"] = peak_r

                    # Check Breakeven Shift
                    if use_be and (not is_be_active) and peak_r >= be_trigger_r:
                        new_sl = entry_p - (be_lock_r * r_dist)
                        if new_sl < sl:
                            sl = new_sl
                            active_trade["sl"] = sl
                            active_trade["is_be_active"] = True

                    # Check MFE Giveback Exit
                    giveback_exit_triggered = False
                    giveback_exit_price = bar_close
                    if use_giveback_exit and peak_r >= giveback_activation_r:
                        peak_profit_dist = entry_p - peak_favorable_price
                        giveback_floor = peak_favorable_price + (peak_profit_dist * giveback_threshold_pct)
                        if bar_high >= giveback_floor:
                            giveback_exit_triggered = True
                            giveback_exit_price = max(bar_open, giveback_floor)

                    if bar_high >= sl:
                        exit_p = sl + slip
                        raw_pnl = (entry_p - exit_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        res_label = "BREAKEVEN" if is_be_active and net_pnl >= 0 else "STOP_LOSS"
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": res_label, "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

                    elif giveback_exit_triggered:
                        exit_p = giveback_exit_price + slip
                        raw_pnl = (entry_p - exit_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": "MFE_GIVEBACK_EXIT", "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

                    elif tp is not None and bar_low <= tp:
                        exit_p = tp + slip
                        raw_pnl = (entry_p - exit_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_pnl = raw_pnl - comm
                        current_equity += net_pnl
                        trades.append({
                            **active_trade, "exit_price": exit_p, "exit_time": bar_time,
                            "pnl": round(net_pnl, 2), "result": "TAKE_PROFIT", "holding_bars": i - active_trade["entry_bar"]
                        })
                        last_exit_direction = direction
                        last_exit_bar = i
                        active_trade = None

            # Track Drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = peak_equity - current_equity
            if dd > max_dd_usd:
                max_dd_usd = dd

            # -------------------------------------------------------------
            # 2. SIGNAL GENERATION & RE-ENTRY
            # -------------------------------------------------------------
            if active_trade is None and in_session and atr > 0.00005:
                # Signals by Strategy Mode
                buy_signal = False
                sell_signal = False
                is_reentry = False

                if strategy_mode == "strat1":
                    # Dual EMA Pullback + RSI
                    is_bull = (ema20s[i] > ema50s[i]) and (bar_close > ema100s[i])
                    is_bear = (ema20s[i] < ema50s[i]) and (bar_close < ema100s[i])
                    pullback_buy = min(lows[i-2:i+1]) <= (ema20s[i] + 0.75 * atr)
                    pullback_sell = max(highs[i-2:i+1]) >= (ema20s[i] - 0.75 * atr)
                    rsi_buy = 40.0 <= rsis[i] <= 62.0
                    rsi_sell = 38.0 <= rsis[i] <= 60.0
                    trig_buy = (bar_close > bar_open) and (bar_close > ema20s[i])
                    trig_sell = (bar_close < bar_open) and (bar_close < ema20s[i])

                    buy_signal = is_bull and pullback_buy and rsi_buy and trig_buy
                    sell_signal = is_bear and pullback_sell and rsi_sell and trig_sell

                elif strategy_mode == "strat4":
                    # Adaptive SuperSmoother
                    buy_signal = ss_longs[i] and (bar_close > ema200s[i]) and (ss_effs[i] >= 0.15)
                    sell_signal = ss_shorts[i] and (bar_close < ema200s[i]) and (ss_effs[i] >= 0.15)

                elif strategy_mode == "hybrid":
                    # Institutional Hybrid: SuperSmoother Regime + 200 EMA Macro + EMA Pullback
                    regime_bull = (ss_sqs[i] == 1) and (bar_close > ema200s[i])
                    regime_bear = (ss_sqs[i] == -1) and (bar_close < ema200s[i])
                    pullback_buy = min(lows[i-2:i+1]) <= (ema20s[i] + 0.5 * atr)
                    pullback_sell = max(highs[i-2:i+1]) >= (ema20s[i] - 0.5 * atr)
                    trig_buy = (bar_close > bar_open) and (bar_close > ema20s[i])
                    trig_sell = (bar_close < bar_open) and (bar_close < ema20s[i])

                    buy_signal = regime_bull and pullback_buy and trig_buy and (ss_effs[i] >= 0.15)
                    sell_signal = regime_bear and pullback_sell and trig_sell and (ss_effs[i] >= 0.15)

                # Check Re-entry Condition
                # If recently exited in profit/giveback within last 8 bars, and trend continues in same direction
                if allow_reentry and (i - last_exit_bar <= 8) and (consecutive_reentries < max_reentries):
                    if last_exit_direction == "BUY" and (ema20s[i] > ema50s[i]) and (bar_close > ema20s[i]) and (bar_close > bar_open):
                        buy_signal = True
                        is_reentry = True
                    elif last_exit_direction == "SELL" and (ema20s[i] < ema50s[i]) and (bar_close < ema20s[i]) and (bar_close < bar_open):
                        sell_signal = True
                        is_reentry = True

                if buy_signal:
                    stop_dist = self.atr_sl_mult * atr
                    tp_dist = self.atr_tp_mult * atr
                    risk_usd = current_equity * (self.risk_pct / 100.0)
                    units = max(1000.0, float(int(risk_usd / stop_dist)))
                    entry_p = bar_close + slip
                    sl = round(entry_p - stop_dist, 5)
                    tp = round(entry_p + tp_dist, 5)

                    active_trade = {
                        "instrument": self.symbol,
                        "direction": "BUY",
                        "entry_time": bar_time,
                        "entry_bar": i,
                        "entry_price": entry_p,
                        "units": units,
                        "sl": sl,
                        "tp": tp,
                        "r_dist": stop_dist,
                        "peak_favorable_price": entry_p,
                        "peak_r": 0.0,
                        "is_be_active": False,
                        "is_reentry": is_reentry
                    }
                    if is_reentry:
                        consecutive_reentries += 1
                    else:
                        consecutive_reentries = 0

                elif sell_signal:
                    stop_dist = self.atr_sl_mult * atr
                    tp_dist = self.atr_tp_mult * atr
                    risk_usd = current_equity * (self.risk_pct / 100.0)
                    units = max(1000.0, float(int(risk_usd / stop_dist)))
                    entry_p = bar_close - slip
                    sl = round(entry_p + stop_dist, 5)
                    tp = round(entry_p - tp_dist, 5)

                    active_trade = {
                        "instrument": self.symbol,
                        "direction": "SELL",
                        "entry_time": bar_time,
                        "entry_bar": i,
                        "entry_price": entry_p,
                        "units": units,
                        "sl": sl,
                        "tp": tp,
                        "r_dist": stop_dist,
                        "peak_favorable_price": entry_p,
                        "peak_r": 0.0,
                        "is_be_active": False,
                        "is_reentry": is_reentry
                    }
                    if is_reentry:
                        consecutive_reentries += 1
                    else:
                        consecutive_reentries = 0

        # Compile Results
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        be_trades = [t for t in trades if t["result"] == "BREAKEVEN"]
        giveback_wins = [t for t in trades if t["result"] == "MFE_GIVEBACK_EXIT"]

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        net_pnl = current_equity - self.capital
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

        return {
            "symbol": self.symbol,
            "strategy": strategy_mode,
            "net_profit": round(net_pnl, 2),
            "return_pct": round((net_pnl / self.capital) * 100.0, 1),
            "profit_factor": round(pf, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "be_count": len(be_trades),
            "giveback_wins": len(giveback_wins),
            "max_dd_usd": round(max_dd_usd, 2),
            "max_dd_pct": round((max_dd_usd / self.capital) * 100.0, 1),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
            "avg_loss": round(gross_loss / len(losses), 2) if losses else 0.0
        }


def run_comparative_experiment():
    print("=" * 90)
    print(" 🚀 COMPARATIVE MATRIX: BASELINE vs DYNAMIC BREAKEVEN vs MFE PROFIT GIVEBACK EXIT")
    print("=" * 90)

    symbols = ["GBPUSD", "EURUSD", "USDJPY"]

    for sym in symbols:
        print(f"\n>>> TESTING {sym} (4-YEAR INSTITUTIONAL MT5 DATA) <<<")
        bt = InstitutionalMFEBacktester(symbol=sym, bars_count=100000)
        df = bt.fetch_mt5_data()
        df = bt.compute_indicators(df)

        models = [
            ("1. Strat 1 Baseline (Fixed SL/TP, No BE/Trailing)", "strat1", False, False, False),
            ("2. Strat 1 + Breakeven (+0.1R @ +1.0R MFE)", "strat1", True, False, False),
            ("3. Strat 1 + Breakeven + MFE Giveback Exit (Banks profit if drops 30%)", "strat1", True, True, False),
            ("4. Strat 1 + Breakeven + MFE Giveback + Smart Re-entry", "strat1", True, True, True),
            ("5. Strat 4 Hybrid + Breakeven + MFE Giveback Exit", "hybrid", True, True, False),
            ("6. Strat 4 Hybrid + Breakeven + MFE Giveback + Smart Re-entry", "hybrid", True, True, True),
        ]

        for label, smode, use_be, use_gb, use_re in models:
            res = bt.run_backtest(
                df,
                strategy_mode=smode,
                use_be=use_be,
                use_giveback_exit=use_gb,
                allow_reentry=use_re
            )
            print(f" {label}:")
            print(f"   Net P&L: ${res['net_profit']:+,.2f} ({res['return_pct']:+.1f}%) | PF: {res['profit_factor']:.2f} | Win Rate: {res['win_rate']}% ({res['wins']}W / {res['losses']}L | {res['be_count']} BE | {res['giveback_wins']} Giveback Exits)")
            print(f"   Max DD: -${res['max_dd_usd']:,.2f} ({res['max_dd_pct']:.1f}%) | Avg Win: ${res['avg_win']:.2f} | Avg Loss: ${res['avg_loss']:.2f}")
            print()


if __name__ == "__main__":
    run_comparative_experiment()
