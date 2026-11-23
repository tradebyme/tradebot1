import os
import sys
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure stategy4 is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import AdaptiveSuperSmootherStrategy
from oanda_api import OandaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester4")


class AdaptiveSmootherBacktester:
    """
    Historical Backtester for Strategy 4: Adaptive SuperSmoother Strategy.
    100% faithful to the Pine Script strategy wrapper:
    - tradeDir: "Long & Short", "Long Only", "Short Only"
    - useSL: True (default) with atrSL = 2.0
    - useTP: False (default) with atrTP = 3.0 (rides trend until opposite flip or SL)
    - sizeMode: "Risk % of Equity" or "Fixed Units"
    - Position Reversal on trend flips
    """

    def __init__(
        self,
        symbol: str = "GBP_USD",
        granularity: str = "M15",
        capital: float = 10000.0,
        trade_dir: str = "Long & Short",
        use_sl: bool = True,
        use_tp: bool = False,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        size_mode: str = "Risk % of Equity",
        risk_pct: float = 1.0,
        fixed_qty: float = 10000.0
    ):
        self.symbol = symbol.replace("/", "_").replace("-", "_")
        if "_" not in self.symbol and len(self.symbol) == 6:
            self.symbol = f"{self.symbol[:3]}_{self.symbol[3:]}"
        self.granularity = granularity
        self.capital = capital
        self.trade_dir = trade_dir
        self.use_sl = use_sl
        self.use_tp = use_tp
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.size_mode = size_mode
        self.risk_pct = risk_pct
        self.fixed_qty = fixed_qty
        self.strategy = AdaptiveSuperSmootherStrategy()

    def fetch_oanda_candles(self, count: int = 500) -> pd.DataFrame:
        """Fetch historical candlestick bars from OANDA API."""
        client = OandaClient()
        logger.info(f"Fetching {count} {self.granularity} candles for {self.symbol} from OANDA...")
        raw_candles = client.get_candles(self.symbol, granularity=self.granularity, count=count)
        if not raw_candles:
            raise ValueError(f"No candles returned from OANDA for {self.symbol}.")

        df = self.strategy.parse_candles(raw_candles)
        df = df[df["complete"] == True].reset_index(drop=True)
        return df

    def run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run bar-by-bar backtest simulation mirroring Pine Script strategy orders."""
        df = self.strategy.calculate_indicators(df)

        n = len(df)
        if n < 50:
            return {"error": "Insufficient bars for backtesting."}

        times = df["time"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atr_vals = df["atr_14"].values
        long_sigs = df["long_signal"].values
        short_sigs = df["short_signal"].values

        pip_unit = 0.01 if "JPY" in self.symbol else (0.10 if "XAU" in self.symbol else 0.0001)

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None
        current_equity = self.capital
        peak_equity = self.capital
        max_drawdown_usd = 0.0
        equity_curve = [{"time": str(times[0]), "equity": self.capital}]

        allow_long = self.trade_dir != "Short Only"
        allow_short = self.trade_dir != "Long Only"

        for i in range(50, n):
            bar_time = str(times[i])
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            curr_atr = atr_vals[i]
            is_long_sig = bool(long_sigs[i])
            is_short_sig = bool(short_sigs[i])

            # 1. Manage Active Trade Exits (Stop Loss / Take Profit)
            if active_trade is not None:
                trade_dir = active_trade["direction"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                units = active_trade["units"]
                entry_p = active_trade["entry_price"]

                # Long checks
                if trade_dir == "BUY":
                    if self.use_sl and sl is not None and bar_low <= sl:
                        pnl = (sl - entry_p) * units
                        active_trade["exit_price"] = sl
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = round(pnl, 2)
                        active_trade["result"] = "STOP_LOSS"
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None
                    elif self.use_tp and tp is not None and bar_high >= tp:
                        pnl = (tp - entry_p) * units
                        active_trade["exit_price"] = tp
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = round(pnl, 2)
                        active_trade["result"] = "TAKE_PROFIT"
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None

                # Short checks
                elif trade_dir == "SELL":
                    if self.use_sl and sl is not None and bar_high >= sl:
                        pnl = (entry_p - sl) * units
                        active_trade["exit_price"] = sl
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = round(pnl, 2)
                        active_trade["result"] = "STOP_LOSS"
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None
                    elif self.use_tp and tp is not None and bar_low <= tp:
                        pnl = (entry_p - tp) * units
                        active_trade["exit_price"] = tp
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = round(pnl, 2)
                        active_trade["result"] = "TAKE_PROFIT"
                        current_equity += pnl
                        trades.append(active_trade)
                        active_trade = None

            # 2. Process Regime Flips (Trend Reversal or New Entries)
            # Signal confirmed on bar close (enters at bar_close / next open)
            if (is_long_sig or is_short_sig) and pd.notna(curr_atr) and curr_atr > 0.00005:
                stop_dist = max(self.atr_sl_mult * curr_atr, 10.0 * pip_unit)
                tp_dist = max(self.atr_tp_mult * curr_atr, 10.0 * pip_unit)

                # Position size calculation (Pine Script sizing formula)
                if self.size_mode == "Fixed Units":
                    qty = self.fixed_qty
                else:
                    risk_amount = current_equity * (self.risk_pct / 100.0)
                    qty = max(1000.0, float(int(risk_amount / stop_dist)))

                if is_long_sig:
                    # If holding a short position, close it (trend flip reversal)
                    if active_trade is not None:
                        if active_trade["direction"] == "SELL":
                            pnl = (active_trade["entry_price"] - bar_close) * active_trade["units"]
                            active_trade["exit_price"] = bar_close
                            active_trade["exit_time"] = bar_time
                            active_trade["pnl"] = round(pnl, 2)
                            active_trade["result"] = "TREND_FLIP_WIN" if pnl > 0 else "TREND_FLIP_LOSS"
                            current_equity += pnl
                            trades.append(active_trade)
                            active_trade = None

                    # Open new long if allowed
                    if allow_long and active_trade is None:
                        sl = round(bar_close - stop_dist, 5) if self.use_sl else None
                        tp = round(bar_close + tp_dist, 5) if self.use_tp else None
                        active_trade = {
                            "instrument": self.symbol,
                            "direction": "BUY",
                            "entry_time": bar_time,
                            "entry_price": bar_close,
                            "units": qty,
                            "sl": sl,
                            "tp": tp,
                            "entry_atr": curr_atr
                        }

                elif is_short_sig:
                    # If holding a long position, close it (trend flip reversal)
                    if active_trade is not None:
                        if active_trade["direction"] == "BUY":
                            pnl = (bar_close - active_trade["entry_price"]) * active_trade["units"]
                            active_trade["exit_price"] = bar_close
                            active_trade["exit_time"] = bar_time
                            active_trade["pnl"] = round(pnl, 2)
                            active_trade["result"] = "TREND_FLIP_WIN" if pnl > 0 else "TREND_FLIP_LOSS"
                            current_equity += pnl
                            trades.append(active_trade)
                            active_trade = None

                    # Open new short if allowed
                    if allow_short and active_trade is None:
                        sl = round(bar_close + stop_dist, 5) if self.use_sl else None
                        tp = round(bar_close - tp_dist, 5) if self.use_tp else None
                        active_trade = {
                            "instrument": self.symbol,
                            "direction": "SELL",
                            "entry_time": bar_time,
                            "entry_price": bar_close,
                            "units": qty,
                            "sl": sl,
                            "tp": tp,
                            "entry_atr": curr_atr
                        }

            # Track Drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = peak_equity - current_equity
            if dd > max_drawdown_usd:
                max_drawdown_usd = dd

            equity_curve.append({"time": bar_time, "equity": round(current_equity, 2)})

        # Close open trade at the end of the data
        if active_trade is not None:
            final_pnl = (
                (closes[-1] - active_trade["entry_price"]) * active_trade["units"]
                if active_trade["direction"] == "BUY"
                else (active_trade["entry_price"] - closes[-1]) * active_trade["units"]
            )
            active_trade["exit_price"] = closes[-1]
            active_trade["exit_time"] = str(times[-1])
            active_trade["pnl"] = round(final_pnl, 2)
            active_trade["result"] = "END_OF_DATA"
            current_equity += final_pnl
            trades.append(active_trade)

        # Statistics
        total_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        net_pl = current_equity - self.capital
        return_pct = (net_pl / self.capital) * 100.0
        max_dd_pct = (max_drawdown_usd / self.capital) * 100.0

        return {
            "symbol": self.symbol,
            "granularity": self.granularity,
            "bars_tested": n,
            "starting_capital": self.capital,
            "ending_equity": round(current_equity, 2),
            "net_profit_usd": round(net_pl, 2),
            "return_pct": round(return_pct, 2),
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "max_drawdown_usd": round(max_drawdown_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "use_sl": self.use_sl,
            "use_tp": self.use_tp,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "trades": trades
        }


def print_report(res: Dict[str, Any]):
    """Print an executive performance table report."""
    print("\n" + "=" * 68)
    print(" 🚀 STRATEGY 4: ADAPTIVE SUPERSMOOTHER (PINE SCRIPT STRATEGY)")
    print("=" * 68)
    print(f" Symbol:               {res['symbol']} ({res['granularity']})")
    print(f" Bars Processed:       {res['bars_tested']:,}")
    print(f" Settings:             useSL={res['use_sl']} ({res['atr_sl_mult']}x) | useTP={res['use_tp']} ({res['atr_tp_mult']}x)")
    print(f" Starting Capital:     ${res['starting_capital']:,.2f}")
    print(f" Ending Equity:        ${res['ending_equity']:,.2f}")
    print(f" Net Profit / Loss:    ${res['net_profit_usd']:+,.2f} ({res['return_pct']:+.2f}%)")
    print("-" * 68)
    print(f" Total Trades:         {res['total_trades']}")
    print(f" Win Rate:             {res['win_rate_pct']:.1f}% ({res['wins']}W / {res['losses']}L)")
    print(f" Profit Factor:        {res['profit_factor']:.2f}")
    print(f" Gross Profit:         +${res['gross_profit']:,.2f}")
    print(f" Gross Loss:           -${res['gross_loss']:,.2f}")
    print(f" Max Drawdown:         -${res['max_drawdown_usd']:,.2f} ({res['max_drawdown_pct']:.2f}%)")
    print("=" * 68)

    if res['total_trades'] > 0:
        print("\nLast 5 Strategy Trades:")
        for t in res['trades'][-5:]:
            print(
                f"  [{t.get('entry_time')[:19]}] {t['direction']} {t['units']:.0f}u @ {t['entry_price']:.5f} -> "
                f"Exit [{t.get('exit_time')[:19]}]: {t.get('exit_price', 0.0):.5f} | "
                f"PnL: ${t.get('pnl', 0.0):+,.2f} ({t.get('result')})"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive SuperSmoother Pine Script Strategy Backtester")
    parser.add_argument("--symbol", type=str, default="GBP_USD", help="Currency pair (e.g. GBP_USD, EUR_USD)")
    parser.add_argument("--count", type=int, default=500, help="Number of historical candles to test")
    parser.add_argument("--dir", type=str, default="Long & Short", choices=["Long & Short", "Long Only", "Short Only"])
    parser.add_argument("--use-sl", action="store_true", default=True, help="Use ATR Stop Loss")
    parser.add_argument("--no-sl", dest="use_sl", action="store_false")
    parser.add_argument("--use-tp", action="store_true", default=False, help="Use ATR Take Profit")
    parser.add_argument("--sl-mult", type=float, default=2.0, help="Stop Loss ATR multiplier")
    parser.add_argument("--tp-mult", type=float, default=3.0, help="Take Profit ATR multiplier")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital (default $10,000)")
    parser.add_argument("--size-mode", type=str, default="Risk % of Equity", choices=["Fixed Units", "Risk % of Equity"])
    args = parser.parse_args()

    backtester = AdaptiveSmootherBacktester(
        symbol=args.symbol,
        capital=args.capital,
        trade_dir=args.dir,
        use_sl=args.use_sl,
        use_tp=args.use_tp,
        atr_sl_mult=args.sl_mult,
        atr_tp_mult=args.tp_mult,
        size_mode=args.size_mode
    )
    df = backtester.fetch_oanda_candles(count=args.count)
    results = backtester.run_simulation(df)
    print_report(results)
