import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Research")

def run():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=730)
    print(f"Fetching 730 days GBPUSD M15...")
    rates = mt5.copy_rates_range("GBPUSD", mt5.TIMEFRAME_M15, utc_from, utc_to)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    print(f"Loaded {n} candles.")

    # 1. ALMA & Ribbon indicators
    alma = ConfluenceTrendStrategy.calculate_alma(df["close"], Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA).values
    dev = df["close"].rolling(window=Config.DEV_LENGTH).std().values
    atr = ConfluenceTrendStrategy.calculate_atr(df["high"], df["low"], df["close"], Config.ATR_LENGTH).values

    slope_score = np.where(atr > 0, (alma - np.roll(alma, Config.SLOPE_LENGTH)) / atr, 0.0)
    slope_score[:Config.SLOPE_LENGTH] = 0.0
    upper_confirm = alma + dev * Config.DEV_MULT
    lower_confirm = alma - dev * Config.DEV_MULT

    bull_setup = (slope_score > Config.SLOPE_MIN) & (close > upper_confirm)
    bear_setup = (slope_score < -Config.SLOPE_MIN) & (close < lower_confirm)

    # Stateful Trend Target Ribbon
    ribbon_trend = np.zeros(n, dtype=int)
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
        ribbon_trend[i] = cur_t

    # 2. Adaptive SuperSmoother
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
    ss_long_flip = np.zeros(n, dtype=bool)
    ss_short_flip = np.zeros(n, dtype=bool)

    cur_sq = 0
    for i in range(1, n):
        if ss_long[i]:
            if cur_sq != 1:
                ss_long_flip[i] = True
            cur_sq = 1
        elif ss_short[i]:
            if cur_sq != -1:
                ss_short_flip[i] = True
            cur_sq = -1
        ss_regime[i] = cur_sq

    print(f"Total Ribbon Bull Flips: {ribbon_bull_flip.sum()}, Bear Flips: {ribbon_bear_flip.sum()}")
    print(f"Total SS Long Flips: {ss_long_flip.sum()}, Short Flips: {ss_short_flip.sum()}")

    # Test several confluence strategies
    def simulate(buy_mask, sell_mask, name):
        trades = []
        active_trade = None
        daily_losses = {}
        lookback = 12

        for i in range(Config.ALMA_LENGTH + 25, n):
            bar_time = df["time"].iloc[i]
            day_str = bar_time.strftime("%Y-%m-%d")
            hour = bar_time.hour
            session_ok = (Config.SESSION_START_HOUR_UTC <= hour < Config.SESSION_END_HOUR_UTC)

            # Exit check
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
                    trades.append(pnl)
                    if exit_res == "LOSS":
                        t_day = active_trade["day"]
                        daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                    active_trade = None

            # Entry check
            if active_trade is None and session_ok:
                if daily_losses.get(day_str, 0) >= 2:
                    continue

                curr_c = close[i]
                c_atr = atr[i]
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
                        "tp": curr_c + (2.0 * risk_dist),
                        "day": day_str
                    }
                elif sell_mask[i]:
                    recent_high = np.max(high[max(0, i - lookback):i])
                    raw_risk = recent_high - curr_c
                    risk_dist = np.clip(raw_risk, c_atr * Config.MIN_STOP_ATR, c_atr * Config.MAX_STOP_ATR)
                    active_trade = {
                        "dir": "SELL",
                        "entry": curr_c,
                        "sl": curr_c + risk_dist,
                        "tp": curr_c - (2.0 * risk_dist),
                        "day": day_str
                    }

        trades = np.array(trades)
        cnt = len(trades)
        if cnt == 0:
            print(f"[{name}] Trades: 0")
            return
        wins = np.sum(trades > 0)
        losses = np.sum(trades < 0)
        wr = (wins / cnt) * 100
        net = np.sum(trades)
        gross_win = np.sum(trades[trades > 0])
        gross_loss = abs(np.sum(trades[trades < 0]))
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        print(f"[{name}] Trades: {cnt}, WinRate: {wr:.1f}% ({wins}W/{losses}L), NetPnL: ${net:+.2f}, ProfitFactor: {pf:.2f}")

    # Mode 1: Ribbon Flip when SS agrees (Ribbon Flip + SS == 1)
    m1_buy = ribbon_bull_flip & (ss_regime == 1)
    m1_sell = ribbon_bear_flip & (ss_regime == -1)
    simulate(m1_buy, m1_sell, "1. Ribbon Flip + SS Reg Aligned")

    # Mode 2: SS Flip when Ribbon agrees (SS Flip + Ribbon == 1)
    m2_buy = ss_long_flip & (ribbon_trend == 1)
    m2_sell = ss_short_flip & (ribbon_trend == -1)
    simulate(m2_buy, m2_sell, "2. SS Flip + Ribbon Reg Aligned")

    # Mode 3: Pure Ribbon Flip (Stand-alone BOSWaves original)
    m3_buy = ribbon_bull_flip
    m3_sell = ribbon_bear_flip
    simulate(m3_buy, m3_sell, "3. Pure BOSWaves Ribbon Flip")

    # Mode 4: Pure SuperSmoother Flip (Stand-alone Ehlers original)
    m4_buy = ss_long_flip
    m4_sell = ss_short_flip
    simulate(m4_buy, m4_sell, "4. Pure Ehlers SS Flip")

    # Mode 5: Trend Pullback to ALMA Bounce (Trend confirmed, price touches/dips ALMA and closes back up)
    # Price dips to ALMA core or lower band, and closes higher while trend == 1 and ss_regime == 1
    dip_buy = (ribbon_trend == 1) & (ss_regime == 1) & (low <= alma) & (close > alma) & (close > df["open"].values)
    dip_sell = (ribbon_trend == -1) & (ss_regime == -1) & (high >= alma) & (close < alma) & (close < df["open"].values)
    simulate(dip_buy, dip_sell, "5. Trend Pullback Bounce to ALMA")

if __name__ == "__main__":
    run()

