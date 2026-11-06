import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")

class Backtester:
    def __init__(self, symbol: str = "GBPUSD", days: int = 730):
        self.symbol = symbol
        self.days = days
        self.capital = 2000.0
        self.target_profit_usd = 160.0       # +8% ($160)
        self.max_drawdown_usd = 160.0       # -8% ($160)
        self.daily_loss_limit_usd = 100.0   # -5% ($100)
        self.circuit_breaker_usd = 70.0     # -3.5% ($70)
        self.risk_per_trade_usd = 25.0      # 1.25% ($25)
        self.risk_reward_ratio = 2.0        # 1:2 R:R (Win = +$50, Loss = -$25)

    def fetch_data(self) -> pd.DataFrame:
        """Fetch up to 2 years of M15 candles from MetaTrader 5."""
        if not mt5.initialize():
            raise RuntimeError(f"MetaTrader 5 initialization failed: {mt5.last_error()}")

        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=self.days)
        logger.info(f"Fetching {self.days} days (~50,000 M15 bars) for {self.symbol} from MetaTrader 5...")
        
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates returned for {self.symbol}.")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate exact indicators as implemented in strategy.py."""
        # EMAs
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

        return df

    def run_full_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized-accelerated bar-by-bar simulation for 50,000+ bars."""
        df = self.calculate_indicators(df)

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

        trades = []
        active_trade = None
        last_trade_day = None
        daily_trades_count = 0
        daily_pnl = 0.0
        circuit_breaker_active = False

        n = len(df)
        for i in range(100, n):
            bar_time = times[i]
            bar_date = bar_time.date()
            bar_hour = bar_time.hour
            weekday = bar_time.weekday()

            # Daily rollover check
            if bar_date != last_trade_day:
                last_trade_day = bar_date
                daily_trades_count = 0
                daily_pnl = 0.0
                circuit_breaker_active = False

            # 1. Manage Active Trade
            if active_trade is not None:
                curr_low = lows[i]
                curr_high = highs[i]

                if active_trade["direction"] == "BUY":
                    if curr_low <= active_trade["sl"]:
                        active_trade["exit_price"] = active_trade["sl"]
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = -self.risk_per_trade_usd
                        active_trade["result"] = "LOSS"
                        daily_pnl += active_trade["pnl"]
                        trades.append(active_trade)
                        active_trade = None
                    elif curr_high >= active_trade["tp"]:
                        active_trade["exit_price"] = active_trade["tp"]
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = self.risk_per_trade_usd * self.risk_reward_ratio
                        active_trade["result"] = "WIN"
                        daily_pnl += active_trade["pnl"]
                        trades.append(active_trade)
                        active_trade = None

                elif active_trade["direction"] == "SELL":
                    if curr_high >= active_trade["sl"]:
                        active_trade["exit_price"] = active_trade["sl"]
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = -self.risk_per_trade_usd
                        active_trade["result"] = "LOSS"
                        daily_pnl += active_trade["pnl"]
                        trades.append(active_trade)
                        active_trade = None
                    elif curr_low <= active_trade["tp"]:
                        active_trade["exit_price"] = active_trade["tp"]
                        active_trade["exit_time"] = bar_time
                        active_trade["pnl"] = self.risk_per_trade_usd * self.risk_reward_ratio
                        active_trade["result"] = "WIN"
                        daily_pnl += active_trade["pnl"]
                        trades.append(active_trade)
                        active_trade = None

                # Check if daily circuit breaker tripped after loss
                if daily_pnl <= -self.circuit_breaker_usd:
                    circuit_breaker_active = True

            if active_trade is not None:
                continue

            # 2. Filters
            # Session filter (07:00 to 17:00 UTC)
            if not (7 <= bar_hour < 17):
                continue

            # Weekend filter
            if weekday == 4 and bar_hour >= 21:
                continue
            if weekday in (5, 6):
                continue

            if circuit_breaker_active:
                continue

            if daily_trades_count >= 3:
                continue

            # 3. Strategy Signals on closed bar (i - 1)
            prev_idx = i - 1
            atr = atr_vals[prev_idx]
            rsi = rsi_vals[prev_idx]
            ema20 = ema20_vals[prev_idx]
            ema50 = ema50_vals[prev_idx]
            ema100 = ema100_vals[prev_idx]
            c_p = closes[prev_idx]
            o_p = opens[prev_idx]

            if np.isnan(atr) or atr <= 0.0001:
                continue

            recent_lows = lows[i-3:i]
            recent_highs = highs[i-3:i]

            # Bullish Setup
            is_bull_trend = (ema20 > ema50) and (c_p > ema100)
            pulled_back_support = any(l <= (ema20 + 0.75 * atr) for l in recent_lows)
            rsi_bull = 40.0 <= rsi <= 62.0
            trigger_bull = (c_p > o_p) and (c_p > ema20)

            if is_bull_trend and pulled_back_support and rsi_bull and trigger_bull:
                entry = opens[i]
                sl = entry - (1.5 * atr)
                tp = entry + (3.0 * atr)
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "BUY",
                    "entry_price": entry,
                    "sl": sl,
                    "tp": tp,
                    "atr": atr,
                    "trade_day": bar_date
                }
                daily_trades_count += 1
                continue

            # Bearish Setup
            is_bear_trend = (ema20 < ema50) and (c_p < ema100)
            pulled_back_resist = any(h >= (ema20 - 0.75 * atr) for h in recent_highs)
            rsi_bear = 38.0 <= rsi <= 60.0
            trigger_bear = (c_p < o_p) and (c_p < ema20)

            if is_bear_trend and pulled_back_resist and rsi_bear and trigger_bear:
                entry = opens[i]
                sl = entry + (1.5 * atr)
                tp = entry - (3.0 * atr)
                active_trade = {
                    "entry_time": bar_time,
                    "direction": "SELL",
                    "entry_price": entry,
                    "sl": sl,
                    "tp": tp,
                    "atr": atr,
                    "trade_day": bar_date
                }
                daily_trades_count += 1
                continue

        trades_df = pd.DataFrame(trades)
        return trades_df

    def evaluate_window_duration(self, trades_df: pd.DataFrame, all_trading_days: list, window_days: int = 3):
        """Evaluate rolling challenge windows of length window_days (e.g. 3, 5, or 10 days)."""
        if len(trades_df) == 0:
            return {}

        trades_df["trade_date"] = pd.to_datetime(trades_df["entry_time"]).dt.date

        total_windows = 0
        target_passed_count = 0
        daily_loss_breached_count = 0
        drawdown_breached_count = 0
        neutral_count = 0

        for idx in range(len(all_trading_days) - (window_days - 1)):
            w_days = all_trading_days[idx:idx + window_days]
            total_windows += 1

            w_trades = trades_df[trades_df["trade_date"].isin(w_days)]
            
            cumulative_pl = 0.0
            daily_pl_map = {d: 0.0 for d in w_days}
            daily_breach = False
            drawdown_breach = False
            target_reached = False

            for _, t in w_trades.iterrows():
                pnl = t["pnl"]
                t_day = t["trade_date"]
                daily_pl_map[t_day] += pnl
                cumulative_pl += pnl

                if daily_pl_map[t_day] <= -self.daily_loss_limit_usd:
                    daily_breach = True
                if cumulative_pl <= -self.max_drawdown_usd:
                    drawdown_breach = True
                if cumulative_pl >= self.target_profit_usd and not daily_breach and not drawdown_breach:
                    target_reached = True

            if target_reached:
                target_passed_count += 1
            elif daily_breach:
                daily_loss_breached_count += 1
            elif drawdown_breach:
                drawdown_breached_count += 1
            else:
                neutral_count += 1

        return {
            "window_days": window_days,
            "total_windows": total_windows,
            "target_passed_count": target_passed_count,
            "target_passed_pct": round((target_passed_count / total_windows) * 100, 1),
            "daily_loss_breached_count": daily_loss_breached_count,
            "daily_loss_breached_pct": round((daily_loss_breached_count / total_windows) * 100, 1),
            "drawdown_breached_count": drawdown_breached_count,
            "drawdown_breached_pct": round((drawdown_breached_count / total_windows) * 100, 1),
            "neutral_count": neutral_count,
            "neutral_pct": round((neutral_count / total_windows) * 100, 1)
        }
