import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure ststegy2 directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtest2")

class Strategy2Backtester:
    """Historical Backtester for Strategy 2 (Adaptive SuperSmoother + Trend Target Ribbon)."""

    def __init__(self, symbol: str = "GBPUSD", days: int = 730):
        self.symbol = symbol
        self.days = days
        self.capital = 2000.0
        self.risk_per_trade_usd = 25.0
        self.risk_reward_ratio = 2.0
        self.max_daily_losses = 2

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical candles from MetaTrader 5 or fallback generator."""
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
            logger.warning(f"MetaTrader 5 fetch failed: {e}. Attempting fallback...")

        # If MT5 is not accessible, check if OANDA or local cache can provide
        raise RuntimeError("Historical data source unavailable. Please ensure MetaTrader 5 is running or provide CSV data.")

    def run_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full historical simulation with exact strategy and risk management rules."""
        logger.info(f"Calculating Strategy 2 indicators on {len(df):,} candles...")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 1. Trend Target Ribbon (ALMA + StDev)
        alma = ConfluenceTrendStrategy.calculate_alma(close, Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA)
        dev = close.rolling(window=Config.DEV_LENGTH).std()
        atr = ConfluenceTrendStrategy.calculate_atr(high, low, close, Config.ATR_LENGTH)

        slope_score = np.where(atr > 0, (alma - alma.shift(Config.SLOPE_LENGTH)) / atr, 0.0)
        upper_confirm = alma + dev * Config.DEV_MULT
        lower_confirm = alma - dev * Config.DEV_MULT

        bull_ribbon = (slope_score > Config.SLOPE_MIN) & (close > upper_confirm)
        bear_ribbon = (slope_score < -Config.SLOPE_MIN) & (close < lower_confirm)

        # 2. Adaptive SuperSmoother
        _, slope, threshold = ConfluenceTrendStrategy.calculate_adaptive_supersmoother(
            close, Config.MA_MIN, Config.MA_MAX, Config.VOL_LENGTH,
            Config.SMOOTH_LENGTH, Config.ER_LENGTH, Config.FAST_BIAS,
            Config.NOISE_LENGTH, Config.NOISE_MULT
        )

        ss_bull = slope > threshold
        ss_bear = slope < -threshold

        df["bull_signal"] = bull_ribbon & ss_bull & (~bull_ribbon.shift(1).fillna(False))
        df["bear_signal"] = bear_ribbon & ss_bear & (~bear_ribbon.shift(1).fillna(False))

        # Simulation state
        trades = []
        active_trade = None
        daily_losses = {}

        lookback = Config.STOP_LOOKBACK
        min_stop_mult = Config.MIN_STOP_ATR
        max_stop_mult = Config.MAX_STOP_ATR

        for i in range(Config.ALMA_LENGTH + 25, len(df)):
            current_bar = df.iloc[i]
            bar_time = current_bar["time"]
            current_day = bar_time.strftime("%Y-%m-%d")
            hour = bar_time.hour

            # Session filter (London 07:00 - NY 17:00 UTC)
            session_ok = (Config.SESSION_START_HOUR_UTC <= hour < Config.SESSION_END_HOUR_UTC)

            # Check active trade exit
            if active_trade is not None:
                trade_dir = active_trade["direction"]
                sl = active_trade["stop_loss"]
                tp = active_trade["take_profit"]

                exit_trade = False
                result = None

                if trade_dir == "BUY":
                    if current_bar["low"] <= sl:
                        exit_trade = True
                        result = "LOSS"
                    elif current_bar["high"] >= tp:
                        exit_trade = True
                        result = "WIN"
                else: # SELL
                    if current_bar["high"] >= sl:
                        exit_trade = True
                        result = "LOSS"
                    elif current_bar["low"] <= tp:
                        exit_trade = True
                        result = "WIN"

                if exit_trade:
                    pnl = 50.0 if result == "WIN" else -25.0
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "direction": trade_dir,
                        "entry_price": active_trade["entry_price"],
                        "exit_price": tp if result == "WIN" else sl,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "result": result,
                        "pnl": pnl,
                        "day": active_trade["day"]
                    })

                    if result == "LOSS":
                        t_day = active_trade["day"]
                        daily_losses[t_day] = daily_losses.get(t_day, 0) + 1

                    active_trade = None

            # Check new entry if no active position
            if active_trade is None and session_ok:
                # Check max 2 daily losses
                if daily_losses.get(current_day, 0) >= self.max_daily_losses:
                    continue

                curr_close = current_bar["close"]
                curr_atr = atr.iloc[i]

                if current_bar["bull_signal"]:
                    recent_low = low.iloc[max(0, i - lookback):i].min()
                    raw_risk = curr_close - recent_low
                    risk_dist = np.clip(raw_risk, curr_atr * min_stop_mult, curr_atr * max_stop_mult)

                    active_trade = {
                        "direction": "BUY",
                        "entry_price": curr_close,
                        "stop_loss": curr_close - risk_dist,
                        "take_profit": curr_close + (self.risk_reward_ratio * risk_dist),
                        "entry_time": bar_time,
                        "day": current_day
                    }

                elif current_bar["bear_signal"]:
                    recent_high = high.iloc[max(0, i - lookback):i].max()
                    raw_risk = recent_high - curr_close
                    risk_dist = np.clip(raw_risk, curr_atr * min_stop_mult, curr_atr * max_stop_mult)

                    active_trade = {
                        "direction": "SELL",
                        "entry_price": curr_close,
                        "stop_loss": curr_close + risk_dist,
                        "take_profit": curr_close - (self.risk_reward_ratio * risk_dist),
                        "entry_time": bar_time,
                        "day": current_day
                    }

        return pd.DataFrame(trades)

if __name__ == "__main__":
    bt = Strategy2Backtester(symbol="GBPUSD", days=730)
    try:
        data = bt.fetch_data()
        results = bt.run_simulation(data)
        print(f"Total Trades: {len(results)}")
        if len(results) > 0:
            wins = len(results[results['result'] == 'WIN'])
            losses = len(results[results['result'] == 'LOSS'])
            print(f"Win Rate: {(wins/len(results))*100:.1f}% ({wins}W / {losses}L)")
            print(f"Net PnL: ${results['pnl'].sum():+,.2f}")
    except Exception as e:
        print(f"Backtest execution error: {e}")
