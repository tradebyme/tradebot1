import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import LiquiditySweepStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMC_Backtester")

class Strategy3Backtester:
    """Historical Backtester for Strategy 3: ICT / SMC Asian Liquidity Sweep & Judas Swing."""

    def __init__(self, symbol: str = "EURUSD", days: int = 730, capital: float = 2000.0, rr_ratio: float = 1.5):
        self.symbol = symbol.replace("_", "")
        self.days = days
        self.capital = capital
        self.risk_per_trade_usd = 25.0
        self.rr_ratio = rr_ratio
        self.max_daily_losses = 2

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical M15 data from MetaTrader 5."""
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                utc_to = datetime.now()
                utc_from = utc_to - timedelta(days=self.days)
                logger.info(f"Fetching {self.days} days M15 data for {self.symbol} from MetaTrader 5...")
                rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
                mt5.shutdown()
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    return df
        except Exception as e:
            logger.warning(f"MetaTrader 5 fetch error: {e}")

        raise RuntimeError(f"Could not load historical data for {self.symbol}. Ensure MT5 is running.")

    def run_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = df["time"].dt.date
        df["hour"] = df["time"].dt.hour
        days = df["date"].unique()

        is_gold = ("XAU" in self.symbol or "GOLD" in self.symbol)
        is_jpy = ("JPY" in self.symbol)
        pip_scale = 0.10 if is_gold else (0.01 if is_jpy else 0.0001)
        sweep_buffer = (5.0 * pip_scale) if is_gold else (Config.SWEEP_BUFFER_PIPS * pip_scale)

        min_range = 30.0 if is_gold else (15.0 if is_jpy else 8.0)
        max_range = 350.0 if is_gold else (90.0 if is_jpy else 55.0)

        trades = []
        daily_losses = {}

        for d in days:
            day_str = str(d)
            if daily_losses.get(day_str, 0) >= self.max_daily_losses:
                continue

            day_df = df[df["date"] == d].copy().reset_index(drop=True)
            # Asian Session: 00:00 to 06:45 UTC
            asian_df = day_df[day_df["hour"] < Config.ASIAN_SESSION_END_UTC]
            if len(asian_df) < 12:
                continue

            asian_high = asian_df["high"].max()
            asian_low = asian_df["low"].min()
            range_pips = (asian_high - asian_low) / pip_scale

            if range_pips < min_range or range_pips > max_range:
                continue

            # London Killzone: 07:00 to 11:00 UTC
            london_df = day_df[(day_df["hour"] >= Config.LONDON_KILLZONE_START_UTC) & (day_df["hour"] < Config.LONDON_KILLZONE_END_UTC)]

            entered = False
            for idx, row in london_df.iterrows():
                if entered:
                    break

                c = row["close"]
                o = row["open"]
                h = row["high"]
                l = row["low"]
                bar_time = row["time"]

                # Bearish Sweep: Pierced Asian High -> Closed back below Asian High -> Bearish Candle (Close < Open)
                if h > asian_high and c < asian_high and c < o:
                    entry = c
                    sl = h + sweep_buffer
                    raw_risk = sl - entry
                    if raw_risk <= 0:
                        continue
                    tp = entry - (self.rr_ratio * raw_risk)
                    entered = True
                    direction = "SELL"
                    start_i = day_df.index[day_df["time"] == bar_time][0]

                # Bullish Sweep: Pierced Asian Low -> Closed back above Asian Low -> Bullish Candle (Close > Open)
                elif l < asian_low and c > asian_low and c > o:
                    entry = c
                    sl = l - sweep_buffer
                    raw_risk = entry - sl
                    if raw_risk <= 0:
                        continue
                    tp = entry + (self.rr_ratio * raw_risk)
                    entered = True
                    direction = "BUY"
                    start_i = day_df.index[day_df["time"] == bar_time][0]

                if entered:
                    # Evaluate outcome through the rest of the day
                    post_df = day_df.iloc[start_i+1:]
                    result = None
                    exit_time = None

                    for _, p_row in post_df.iterrows():
                        if direction == "BUY":
                            if p_row["low"] <= sl:
                                result = "LOSS"
                                exit_time = p_row["time"]
                                break
                            elif p_row["high"] >= tp:
                                result = "WIN"
                                exit_time = p_row["time"]
                                break
                        else:
                            if p_row["high"] >= sl:
                                result = "LOSS"
                                exit_time = p_row["time"]
                                break
                            elif p_row["low"] <= tp:
                                result = "WIN"
                                exit_time = p_row["time"]
                                break

                    # If trade did not hit SL or TP before session close, close at market
                    if result is None:
                        exit_time = day_df["time"].iloc[-1]
                        last_c = day_df["close"].iloc[-1]
                        if direction == "BUY":
                            pnl = ((last_c - entry) / raw_risk) * self.risk_per_trade_usd
                        else:
                            pnl = ((entry - last_c) / raw_risk) * self.risk_per_trade_usd
                        result = "WIN" if pnl > 0 else "LOSS"
                    else:
                        pnl = (self.risk_per_trade_usd * self.rr_ratio) if result == "WIN" else -self.risk_per_trade_usd

                    trades.append({
                        "entry_time": bar_time,
                        "exit_time": exit_time,
                        "direction": direction,
                        "entry_price": entry,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "result": result,
                        "pnl": pnl,
                        "day": day_str
                    })

                    if result == "LOSS":
                        daily_losses[day_str] = daily_losses.get(day_str, 0) + 1

        return pd.DataFrame(trades)

def main():
    parser = argparse.ArgumentParser(description="Strategy 3 (ICT / SMC Liquidity Sweep) Backtester")
    parser.add_argument("--symbol", default="EURUSD", help="Currency pair (e.g. EURUSD, GBPUSD, XAUUSD)")
    parser.add_argument("--days", type=int, default=730, help="Days of history (e.g. 730 = 2 years)")
    parser.add_argument("--rr", type=float, default=1.5, help="Risk to reward ratio (default: 1.5)")
    args = parser.parse_args()

    bt = Strategy3Backtester(symbol=args.symbol, days=args.days, rr_ratio=args.rr)
    data = bt.fetch_data()
    results = bt.run_simulation(data)

    print("\n=======================================================")
    print(f"STRATEGY 3 SMC BACKTEST REPORT: {args.symbol} (2 YEARS / {args.days} DAYS)")
    print(f"Risk-to-Reward: 1:{args.rr:.1f} | Fixed Risk: $25.00 (1.25%)")
    print("=======================================================")

    tot = len(results)
    if tot == 0:
        print("No trades found.")
        return

    wins = len(results[results["result"] == "WIN"])
    losses = len(results[results["result"] == "LOSS"])
    wr = (wins / tot) * 100
    net = results["pnl"].sum()
    gw = results[results["pnl"] > 0]["pnl"].sum()
    gl = abs(results[results["pnl"] < 0]["pnl"].sum())
    pf = (gw / gl) if gl > 0 else 999.0

    results["cum_pnl"] = results["pnl"].cumsum()
    results["equity"] = 2000.0 + results["cum_pnl"]
    peak = results["equity"].cummax()
    max_dd = (peak - results["equity"]).max()
    max_dd_pct = (max_dd / 2000.0) * 100

    print(f"Total Completed Trades:   {tot}")
    print(f"Overall Win Rate:         {wr:.1f}% ({wins} Wins / {losses} Losses)")
    print(f"Starting Capital:         ${bt.capital:,.2f}")
    print(f"Net Profit / PnL:         ${net:+,.2f} ({(net/bt.capital)*100:+.1f}%)")
    print(f"Gross Profit:             ${gw:,.2f}")
    print(f"Gross Loss:               ${gl:,.2f}")
    print(f"Profit Factor:            {pf:.2f}")
    print(f"Max Absolute Drawdown:    ${max_dd:,.2f} ({max_dd_pct:.1f}% of starting capital)")
    print(f"Risk Per Trade:           ${bt.risk_per_trade_usd:.2f}")
    print(f"Reward Per Trade:         ${bt.risk_per_trade_usd * args.rr:.2f}")
    print(f"Daily Circuit Breaker:    Max 2 Losses ($50.00 limit)\n")

if __name__ == "__main__":
    main()
