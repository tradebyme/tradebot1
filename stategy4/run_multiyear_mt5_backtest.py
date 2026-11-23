import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Ensure stategy4 is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy import AdaptiveSuperSmootherStrategy


class MT5MultiYearBacktester:
    """
    Institutional Multi-Year Backtester for Strategy 4: Adaptive SuperSmoother Strategy.
    Fetches real institutional tick/bar data from MetaTrader 5 over 2 to 4 years (up to 100,000 M15 bars).
    """

    def __init__(
        self,
        symbol: str = "GBPUSD",
        bars_count: int = 100000,
        initial_capital: float = 10000.0,
        trade_dir: str = "Long & Short",
        use_sl: bool = True,
        use_tp: bool = False,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        size_mode: str = "Risk % of Equity",
        risk_pct: float = 1.0,
        fixed_qty: float = 10000.0,
        session_filter: bool = False,
        session_start_utc: int = 7,
        session_end_utc: int = 17,
        use_ema_filter: bool = False,
        min_efficiency: float = 0.0,
        commission_pct: float = 0.00002,  # 0.002% per turn (~$2 per lot)
        slippage_pips: float = 0.5
    ):
        self.symbol = symbol.replace("/", "").replace("_", "")
        self.bars_count = bars_count
        self.capital = initial_capital
        self.trade_dir = trade_dir
        self.use_sl = use_sl
        self.use_tp = use_tp
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.size_mode = size_mode
        self.risk_pct = risk_pct
        self.fixed_qty = fixed_qty
        self.session_filter = session_filter
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.use_ema_filter = use_ema_filter
        self.min_efficiency = min_efficiency
        self.commission_pct = commission_pct
        self.slippage_pips = slippage_pips
        self.pip_unit = 0.01 if "JPY" in self.symbol else (0.10 if "XAU" in self.symbol else 0.0001)

    def fetch_mt5_data(self) -> pd.DataFrame:
        """Fetch historical candlestick data from MetaTrader 5 terminal."""
        if not mt5.initialize():
            raise RuntimeError(f"MetaTrader 5 initialization failed: {mt5.last_error()}")

        print(f"Fetching up to {self.bars_count:,} M15 bars for {self.symbol} from MetaTrader 5...")
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, self.bars_count)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates returned from MetaTrader 5 for symbol: {self.symbol}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute institutional bar-by-bar backtest simulation."""
        n = len(df)
        if n < 100:
            return {"error": "Insufficient bars"}

        close_series = df["close"]
        high_series = df["high"]
        low_series = df["low"]

        # ATR 14
        prev_close = close_series.shift(1)
        tr1 = high_series - low_series
        tr2 = (high_series - prev_close).abs()
        tr3 = (low_series - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=1).mean().values

        # 200 EMA Macro Trend
        ema200 = close_series.ewm(span=200, adjust=False).mean().values

        # Adaptive SuperSmoother
        smoother_res = AdaptiveSuperSmootherStrategy.calculate_adaptive_supersmoother(close_series)
        long_sigs = smoother_res["long_signal"].values
        short_sigs = smoother_res["short_signal"].values
        eff_vals = smoother_res["efficiency"].values

        times_arr = df["time"].to_numpy()
        opens = df["open"].values
        highs = high_series.values
        lows = low_series.values
        closes = close_series.values

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None
        current_equity = self.capital
        peak_equity = self.capital
        max_drawdown_usd = 0.0

        allow_long = self.trade_dir != "Short Only"
        allow_short = self.trade_dir != "Long Only"

        # Precompute session mask
        hours = pd.to_datetime(times_arr).hour
        if self.session_filter:
            session_mask = (hours >= self.session_start_utc) & (hours < self.session_end_utc)
        else:
            session_mask = np.ones(n, dtype=bool)

        slip = self.slippage_pips * self.pip_unit

        for i in range(100, n):
            bar_time = times_arr[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            curr_atr = atr_14[i]
            curr_ema = ema200[i]
            curr_eff = eff_vals[i]
            is_long_sig = long_sigs[i]
            is_short_sig = short_sigs[i]
            in_session = session_mask[i]

            # 1. Manage Active Trade Exits (Stop Loss / Take Profit)
            if active_trade is not None:
                t_dir = active_trade["direction"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                units = active_trade["units"]
                entry_p = active_trade["entry_price"]

                if t_dir == "BUY":
                    if self.use_sl and sl is not None and bar_low <= sl:
                        exit_p = sl - slip
                        raw_pnl = (exit_p - entry_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "STOP_LOSS",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None
                    elif self.use_tp and tp is not None and bar_high >= tp:
                        exit_p = tp - slip
                        raw_pnl = (exit_p - entry_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "TAKE_PROFIT",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None

                elif t_dir == "SELL":
                    if self.use_sl and sl is not None and bar_high >= sl:
                        exit_p = sl + slip
                        raw_pnl = (entry_p - exit_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "STOP_LOSS",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None
                    elif self.use_tp and tp is not None and bar_low <= tp:
                        exit_p = tp + slip
                        raw_pnl = (entry_p - exit_p) * units
                        comm = (entry_p + exit_p) * units * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "TAKE_PROFIT",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None

            # 2. Process Regime Flips (Trend Reversal or New Entries)
            if (is_long_sig or is_short_sig) and curr_atr > 0.00005:
                stop_dist = max(self.atr_sl_mult * curr_atr, 10.0 * self.pip_unit)
                tp_dist = max(self.atr_tp_mult * curr_atr, 10.0 * self.pip_unit)

                # Determine sizing
                if self.size_mode == "Fixed Units":
                    qty = self.fixed_qty
                else:
                    risk_usd = current_equity * (self.risk_pct / 100.0)
                    qty = max(1000.0, float(int(risk_usd / stop_dist)))

                # Macro filter check
                long_filter_ok = (not self.use_ema_filter) or (bar_close > curr_ema)
                short_filter_ok = (not self.use_ema_filter) or (bar_close < curr_ema)
                eff_ok = curr_eff >= self.min_efficiency

                # LONG SIGNAL
                if is_long_sig:
                    # If holding Short, close it immediately (Trend flip reversal)
                    if active_trade is not None and active_trade["direction"] == "SELL":
                        exit_p = bar_close + slip
                        raw_pnl = (active_trade["entry_price"] - exit_p) * active_trade["units"]
                        comm = (active_trade["entry_price"] + exit_p) * active_trade["units"] * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "TREND_FLIP_WIN" if net_trade_pnl > 0 else "TREND_FLIP_LOSS",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None

                    # Open new Long
                    if allow_long and active_trade is None and in_session and long_filter_ok and eff_ok:
                        entry_p = bar_close + slip
                        sl = round(entry_p - stop_dist, 5) if self.use_sl else None
                        tp = round(entry_p + tp_dist, 5) if self.use_tp else None
                        active_trade = {
                            "instrument": self.symbol,
                            "direction": "BUY",
                            "entry_time": bar_time,
                            "entry_bar": i,
                            "entry_price": entry_p,
                            "units": qty,
                            "sl": sl,
                            "tp": tp,
                            "entry_atr": curr_atr
                        }

                # SHORT SIGNAL
                elif is_short_sig:
                    # If holding Long, close it immediately (Trend flip reversal)
                    if active_trade is not None and active_trade["direction"] == "BUY":
                        exit_p = bar_close - slip
                        raw_pnl = (exit_p - active_trade["entry_price"]) * active_trade["units"]
                        comm = (active_trade["entry_price"] + exit_p) * active_trade["units"] * self.commission_pct
                        net_trade_pnl = raw_pnl - comm
                        active_trade.update({
                            "exit_price": exit_p,
                            "exit_time": bar_time,
                            "pnl": round(net_trade_pnl, 2),
                            "result": "TREND_FLIP_WIN" if net_trade_pnl > 0 else "TREND_FLIP_LOSS",
                            "holding_bars": i - active_trade["entry_bar"]
                        })
                        current_equity += net_trade_pnl
                        trades.append(active_trade)
                        active_trade = None

                    # Open new Short
                    if allow_short and active_trade is None and in_session and short_filter_ok and eff_ok:
                        entry_p = bar_close - slip
                        sl = round(entry_p + stop_dist, 5) if self.use_sl else None
                        tp = round(entry_p - tp_dist, 5) if self.use_tp else None
                        active_trade = {
                            "instrument": self.symbol,
                            "direction": "SELL",
                            "entry_time": bar_time,
                            "entry_bar": i,
                            "entry_price": entry_p,
                            "units": qty,
                            "sl": sl,
                            "tp": tp,
                            "entry_atr": curr_atr
                        }

            # Track peak equity & drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = peak_equity - current_equity
            if dd > max_drawdown_usd:
                max_drawdown_usd = dd

        # Close open trade at end
        if active_trade is not None:
            last_p = closes[-1]
            if active_trade["direction"] == "BUY":
                raw_pnl = (last_p - active_trade["entry_price"]) * active_trade["units"]
            else:
                raw_pnl = (active_trade["entry_price"] - last_p) * active_trade["units"]
            comm = (active_trade["entry_price"] + last_p) * active_trade["units"] * self.commission_pct
            net_trade_pnl = raw_pnl - comm
            active_trade.update({
                "exit_price": last_p,
                "exit_time": times_arr[-1],
                "pnl": round(net_trade_pnl, 2),
                "result": "END_OF_DATA",
                "holding_bars": n - 1 - active_trade["entry_bar"]
            })
            current_equity += net_trade_pnl
            trades.append(active_trade)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return self._compute_analytics(df, trades_df, current_equity, max_drawdown_usd)

    def _compute_analytics(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame,
        ending_equity: float,
        max_drawdown_usd: float
    ) -> Dict[str, Any]:
        """Compute institutional analytics, yearly breakdowns, and prop evaluation windows."""
        n_bars = len(df)
        start_time = df["time"].iloc[0]
        end_time = df["time"].iloc[-1]
        days_covered = (end_time - start_time).days

        if trades_df.empty:
            return {
                "symbol": self.symbol,
                "bars": n_bars,
                "start": start_time.strftime("%Y-%m-%d"),
                "end": end_time.strftime("%Y-%m-%d"),
                "days_covered": days_covered,
                "years_covered": round(days_covered / 365.25, 1),
                "capital": self.capital,
                "ending_equity": self.capital,
                "net_profit": 0.0,
                "return_pct": 0.0,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "payoff_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_holding_hours": 0.0,
                "max_consec_wins": 0,
                "max_consec_losses": 0,
                "yearly_stats": {},
                "challenge_results": {"3d": {"total_windows": 0, "passed": 0, "pass_pct": 0.0}},
                "trades_df": trades_df
            }

        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
        net_profit = ending_equity - self.capital
        return_pct = (net_profit / self.capital) * 100.0
        max_dd_pct = (max_drawdown_usd / self.capital) * 100.0

        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = abs(losses["pnl"].mean()) if not losses.empty else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        avg_holding_bars = trades_df["holding_bars"].mean()
        avg_holding_hours = avg_holding_bars * 15.0 / 60.0

        # Streaks
        pnl_signs = (trades_df["pnl"] > 0).astype(int)
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

        # Yearly breakdown
        trades_df["year"] = pd.to_datetime(trades_df["entry_time"]).dt.year
        yearly_stats = {}
        for yr, group in trades_df.groupby("year"):
            yr_wins = group[group["pnl"] > 0]
            yr_losses = group[group["pnl"] <= 0]
            yr_gp = yr_wins["pnl"].sum()
            yr_gl = abs(yr_losses["pnl"].sum())
            yr_pf = (yr_gp / yr_gl) if yr_gl > 0 else 999.0
            yearly_stats[yr] = {
                "trades": len(group),
                "win_rate": round(len(yr_wins) / len(group) * 100.0, 1),
                "net_profit": round(group["pnl"].sum(), 2),
                "profit_factor": round(yr_pf, 2)
            }

        # Prop challenge rolling windows ($2k baseline capital, +8% target = +$160, -5% daily stop = -$100, -8% max loss = -$160)
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

                if worst_day <= -100.0:  # 5% of $2,000
                    fail_daily_count += 1
                elif w_pnl <= -160.0:    # 8% of $2,000
                    fail_dd_count += 1
                elif w_pnl >= 160.0:     # +8% target
                    pass_count += 1
                else:
                    neutral_count += 1

            challenge_results[f"{window_days}d"] = {
                "total_windows": total_windows,
                "passed": pass_count,
                "pass_pct": round((pass_count / total_windows * 100.0) if total_windows > 0 else 0.0, 1),
                "failed_daily": fail_daily_count,
                "failed_dd": fail_dd_count,
                "neutral": neutral_count
            }

        return {
            "symbol": self.symbol,
            "bars": n_bars,
            "start": start_time.strftime("%Y-%m-%d"),
            "end": end_time.strftime("%Y-%m-%d"),
            "days_covered": days_covered,
            "years_covered": round(days_covered / 365.25, 1),
            "capital": self.capital,
            "ending_equity": round(ending_equity, 2),
            "net_profit": round(net_profit, 2),
            "return_pct": round(return_pct, 2),
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "max_drawdown_usd": round(max_drawdown_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_holding_hours": round(avg_holding_hours, 1),
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "yearly_stats": yearly_stats,
            "challenge_results": challenge_results,
            "trades_df": trades_df
        }


def run_full_multiyear_matrix():
    """Run comprehensive multi-symbol, multi-year backtests across all configurations."""
    symbols = ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD"]

    print("\n" + "=" * 80)
    print(" 🚀 STRATEGY 4: ADAPTIVE SUPERSMOOTHER MULTI-YEAR FOREX BACKTEST (REAL MT5 DATA) ")
    print("=" * 80)

    for sym in symbols:
        print(f"\n==================== [ {sym} ] MULTI-YEAR TESTING ====================")

        # Baseline Data
        bt_base = MT5MultiYearBacktester(symbol=sym, bars_count=100000)
        df = bt_base.fetch_mt5_data()

        # Model A: Pure Trend Riding (Pine Script default: useSL=2.0x, useTP=False)
        bt_a = MT5MultiYearBacktester(
            symbol=sym,
            use_sl=True,
            atr_sl_mult=2.0,
            use_tp=False,
            session_filter=False
        )
        res_a = bt_a.run_simulation(df)

        # Model B: Take Profit (Pine Script: useSL=2.0x, useTP=3.0x ATR)
        bt_b = MT5MultiYearBacktester(
            symbol=sym,
            use_sl=True,
            atr_sl_mult=2.0,
            use_tp=True,
            atr_tp_mult=3.0,
            session_filter=False
        )
        res_b = bt_b.run_simulation(df)

        # Model C: London / NY Session Filter (07:00 - 17:00 UTC) with TP
        bt_c = MT5MultiYearBacktester(
            symbol=sym,
            use_sl=True,
            atr_sl_mult=2.0,
            use_tp=True,
            atr_tp_mult=3.0,
            session_filter=True,
            session_start_utc=7,
            session_end_utc=17
        )
        res_c = bt_c.run_simulation(df)

        # Model D: Institutional Confluence (London/NY + 200 EMA Macro Alignment + Efficiency >= 0.20 + TP)
        bt_d = MT5MultiYearBacktester(
            symbol=sym,
            use_sl=True,
            atr_sl_mult=1.5,
            use_tp=True,
            atr_tp_mult=3.0,
            session_filter=True,
            session_start_utc=7,
            session_end_utc=17,
            use_ema_filter=True,
            min_efficiency=0.20
        )
        res_d = bt_d.run_simulation(df)

        print(f" Period: {res_a['start']} to {res_a['end']} ({res_a['years_covered']} Years, {res_a['bars']:,} Bars)")
        print("-" * 80)

        for label, r in [
            ("MODEL A (Pine Default: Trend Riding, useSL=2.0, useTP=False)", res_a),
            ("MODEL B (Take Profit Target: useSL=2.0, useTP=3.0)", res_b),
            ("MODEL C (London/NY Session 07-17 UTC + Take Profit)", res_c),
            ("MODEL D (Institutional Confluence: Session + 200 EMA + Efficiency >= 0.20)", res_d),
        ]:
            print(f" {label}:")
            print(f"   • Net PnL: ${r['net_profit']:+,.2f} ({r['return_pct']:+.1f}%) | PF: {r['profit_factor']:.2f} | Win Rate: {r['win_rate']}% ({r['wins']}W/{r['losses']}L)")
            print(f"   • Max DD: -${r['max_drawdown_usd']:,.2f} ({r['max_drawdown_pct']:.1f}%) | Payoff: {r['payoff_ratio']:.2f}x (Avg Win: ${r['avg_win']}, Avg Loss: ${r['avg_loss']})")
            ch3 = r['challenge_results'].get('3d', {})
            print(f"   • 3-Day Challenge Pass: {ch3.get('passed', 0)}/{ch3.get('total_windows', 0)} ({ch3.get('pass_pct', 0.0)}%)")
            if r['yearly_stats']:
                yr_strs = [f"{yr}: Net ${stats['net_profit']:+,.0f} (PF {stats['profit_factor']:.2f})" for yr, stats in r['yearly_stats'].items()]
                print(f"   • By Year: {' | '.join(yr_strs)}")
            print()
        print("=" * 80)


if __name__ == "__main__":
    run_full_multiyear_matrix()
