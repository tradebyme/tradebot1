import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")

def fast_alma(series, length=34, offset=0.85, sigma=6.0):
    m = offset * (length - 1)
    s = length / sigma
    indices = np.arange(length)
    weights = np.exp(-((indices - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()
    res = np.full(len(series), np.nan)
    conv = np.convolve(series.values, weights[::-1], mode='valid')
    res[length - 1:] = conv
    return res

class Strategy2Backtester:
    """Institutional Multi-Year Backtester for Strategy 2 (ALMA + Adaptive SuperSmoother + 200 EMA)."""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "H1", days: int = 1095, capital: float = 2000.0):
        self.symbol = symbol.replace("_", "")
        self.timeframe = timeframe.upper()
        self.days = days
        self.capital = capital
        self.risk_per_trade_usd = 25.0
        self.risk_reward_ratio = 2.0
        self.max_daily_losses = 2

    def fetch_data(self) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                tf = mt5.TIMEFRAME_H1 if self.timeframe == "H1" else mt5.TIMEFRAME_M15
                utc_to = datetime.now()
                utc_from = utc_to - timedelta(days=self.days)
                logger.info(f"Fetching {self.days} days {self.timeframe} data for {self.symbol} from MetaTrader 5...")
                rates = mt5.copy_rates_range(self.symbol, tf, utc_from, utc_to)
                mt5.shutdown()
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    return df
        except Exception as e:
            logger.warning(f"MetaTrader 5 fetch failed: {e}")

        raise RuntimeError(f"Could not load historical data for {self.symbol}. Ensure MT5 is running.")

    def run_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        times = df["time"]

        # 1. 200 EMA Higher Trend Filter
        ema200 = df["close"].ewm(span=200, adjust=False).mean().values

        # 2. ALMA & Deviation Ribbon
        alma = fast_alma(df["close"], Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA)
        dev = df["close"].rolling(window=Config.DEV_LENGTH).std().values
        atr = ConfluenceTrendStrategy.calculate_atr(df["high"], df["low"], df["close"], Config.ATR_LENGTH).values

        slope_len = Config.SLOPE_LENGTH
        shift_alma = np.roll(alma, slope_len)
        shift_alma[:slope_len] = np.nan
        slope_score = np.where((atr > 0) & (~np.isnan(alma)) & (~np.isnan(shift_alma)),
                               (alma - shift_alma) / atr, 0.0)

        upper_confirm = alma + dev * Config.DEV_MULT
        lower_confirm = alma - dev * Config.DEV_MULT

        bull_setup = (slope_score > Config.SLOPE_MIN) & (close > upper_confirm)
        bear_setup = (slope_score < -Config.SLOPE_MIN) & (close < lower_confirm)

        ribbon_bull_flip = np.zeros(n, dtype=bool)
        ribbon_bear_flip = np.zeros(n, dtype=bool)
        cur_t = 0
        for i in range(1, n):
            if cur_t != 1 and bull_setup[i]:
                cur_t = 1
                ribbon_bull_flip[i] = True
            elif cur_t != -1 and bear_setup[i]:
                cur_t = -1
                ribbon_bear_flip[i] = True

        # 3. Adaptive SuperSmoother
        trend_ss, slope, threshold = ConfluenceTrendStrategy.calculate_adaptive_supersmoother(
            df["close"], Config.MA_MIN, Config.MA_MAX, Config.VOL_LENGTH,
            Config.SMOOTH_LENGTH, Config.ER_LENGTH, Config.FAST_BIAS,
            Config.NOISE_LENGTH, Config.NOISE_MULT
        )
        slope = slope.values
        threshold = threshold.values
        ss_long = slope > threshold
        ss_short = slope < -threshold

        ss_regime = np.zeros(n, dtype=int)
        cur_sq = 0
        for i in range(1, n):
            if ss_long[i]:
                cur_sq = 1
            elif ss_short[i]:
                cur_sq = -1
            ss_regime[i] = cur_sq

        # Confluence Triggers
        buy_mask = ribbon_bull_flip & (ss_regime == 1) & (close > ema200)
        sell_mask = ribbon_bear_flip & (ss_regime == -1) & (close < ema200)

        trades = []
        active_trade = None
        daily_losses = {}
        lookback = Config.STOP_LOOKBACK

        for i in range(250, n):
            bar_time = times.iloc[i]
            day_str = bar_time.strftime("%Y-%m-%d")
            curr_c = close[i]
            c_atr = atr[i]

            if active_trade is not None:
                tdir = active_trade["dir"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                exit_res = None

                if tdir == "BUY":
                    if low[i] <= sl:
                        exit_res = "LOSS"
                    elif high[i] >= tp:
                        exit_res = "WIN"
                else:
                    if high[i] >= sl:
                        exit_res = "LOSS"
                    elif low[i] <= tp:
                        exit_res = "WIN"

                if exit_res:
                    pnl = 50.0 if exit_res == "WIN" else -25.0
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": tdir,
                        "entry_price": active_trade["entry"],
                        "exit_price": tp if exit_res == "WIN" else sl,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "result": exit_res,
                        "pnl": pnl,
                        "day": active_trade["day"]
                    })
                    if exit_res == "LOSS":
                        t_day = active_trade["day"]
                        daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                    active_trade = None

            if active_trade is None:
                if daily_losses.get(day_str, 0) >= self.max_daily_losses:
                    continue
                if np.isnan(c_atr) or c_atr <= 0:
                    continue

                if buy_mask[i]:
                    recent_low = np.min(low[max(0, i - lookback):i])
                    raw_risk = curr_c - recent_low
                    risk_dist = np.clip(raw_risk, c_atr * Config.MIN_STOP_ATR, c_atr * Config.MAX_STOP_ATR)
                    active_trade = {
                        "dir": "BUY",
                        "entry": curr_c,
                        "sl": curr_c - risk_dist,
                        "tp": curr_c + (self.risk_reward_ratio * risk_dist),
                        "day": day_str,
                        "entry_time": bar_time
                    }
                elif sell_mask[i]:
                    recent_high = np.max(high[max(0, i - lookback):i])
                    raw_risk = recent_high - curr_c
                    risk_dist = np.clip(raw_risk, c_atr * Config.MIN_STOP_ATR, c_atr * Config.MAX_STOP_ATR)
                    active_trade = {
                        "dir": "SELL",
                        "entry": curr_c,
                        "sl": curr_c + risk_dist,
                        "tp": curr_c - (self.risk_reward_ratio * risk_dist),
                        "day": day_str,
                        "entry_time": bar_time
                    }

        return pd.DataFrame(trades)

def main():
    parser = argparse.ArgumentParser(description="Strategy 2 Historical Backtest")
    parser.add_argument("--symbol", default="XAUUSD", help="Forex/Commodity pair (e.g. XAUUSD, EURUSD, USDJPY)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (H1 or M15)")
    parser.add_argument("--days", type=int, default=1095, help="Historical days (e.g. 1095 = 3 years)")
    args = parser.parse_args()

    bt = Strategy2Backtester(symbol=args.symbol, timeframe=args.timeframe, days=args.days)
    data = bt.fetch_data()
    results = bt.run_simulation(data)

    print("\n=======================================================")
    print(f"BACKTEST REPORT: {args.symbol} ({args.timeframe}) - {args.days} DAYS")
    print("=======================================================")
    tot = len(results)
    if tot == 0:
        print("No trades generated.")
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

    print(f"Total Candles Tested:     {len(data):,}")
    print(f"Total Completed Trades:   {tot}")
    print(f"Win Rate:                 {wr:.1f}% ({wins} Wins / {losses} Losses)")
    print(f"Starting Capital:         ${bt.capital:,.2f}")
    print(f"Net Profit / PnL:         ${net:+,.2f} ({(net/bt.capital)*100:+.1f}%)")
    print(f"Gross Profit:             ${gw:,.2f}")
    print(f"Gross Loss:               ${gl:,.2f}")
    print(f"Profit Factor:            {pf:.2f}")
    print(f"Max Absolute Drawdown:    ${max_dd:,.2f} ({max_dd_pct:.1f}% of starting capital)")
    print(f"Risk Per Trade:           ${bt.risk_per_trade_usd:.2f} (1.25%)")
    print(f"Target Per Trade:         ${bt.risk_per_trade_usd * bt.risk_reward_ratio:.2f} (2.50%) [1:2 R:R]")
    print(f"Daily Loss Circuit Breaker: Max 2 Losses (${bt.risk_per_trade_usd * 2:.2f})\n")

if __name__ == "__main__":
    main()
