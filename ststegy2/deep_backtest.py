import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

def get_data(symbol="GBPUSD", days=730):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def fast_alma(series, length=34, offset=0.85, sigma=6.0):
    m = offset * (length - 1)
    s = length / sigma
    indices = np.arange(length)
    weights = np.exp(-((indices - m) ** 2) / (2.0 * s * s))
    weights /= weights.sum()
    # Fast convolution
    res = np.full(len(series), np.nan)
    conv = np.convolve(series.values, weights[::-1], mode='valid')
    res[length - 1:] = conv
    return res

def run_tests():
    symbols = ["GBPUSD", "EURUSD"]
    for sym in symbols:
        print(f"\n==========================================")
        print(f"ANALYZING {sym} (730 Days M15)")
        print(f"==========================================")
        df = get_data(sym, 730)
        n = len(df)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_p = df["open"].values
        times = df["time"]

        # 200 EMA higher trend filter
        ema200 = df["close"].ewm(span=200, adjust=False).mean().values

        # ALMA
        alma = fast_alma(df["close"], Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA)
        dev = df["close"].rolling(window=Config.DEV_LENGTH).std().values
        atr = ConfluenceTrendStrategy.calculate_atr(df["high"], df["low"], df["close"], Config.ATR_LENGTH).values

        slope_len = Config.SLOPE_LENGTH
        slope_score = np.zeros(n)
        valid_idx = ~np.isnan(alma)
        shift_alma = np.roll(alma, slope_len)
        shift_alma[:slope_len] = np.nan
        slope_score = np.where((atr > 0) & (~np.isnan(alma)) & (~np.isnan(shift_alma)),
                               (alma - shift_alma) / atr, 0.0)

        upper_confirm = alma + dev * Config.DEV_MULT
        lower_confirm = alma - dev * Config.DEV_MULT

        bull_setup = (slope_score > Config.SLOPE_MIN) & (close > upper_confirm)
        bear_setup = (slope_score < -Config.SLOPE_MIN) & (close < lower_confirm)

        # Stateful Ribbon Trend
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

        # SuperSmoother
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

        # Let's test combinations:
        # A: Ribbon Flip + 200 EMA Filter
        # B: Ribbon Flip + SuperSmoother Confluence + 200 EMA Filter
        # C: Pullback to ALMA bounce + 200 EMA Filter
        # D: Ribbon Flip with Break-Even Trigger at 1R
        # E: Different R:R targets (1:1.5, 1:2, 1:2.5)

        configs = [
            ("Ribbon Flip (Unfiltered, 1:2)", ribbon_bull_flip, ribbon_bear_flip, 2.0, False, False),
            ("Ribbon Flip + 200 EMA (1:2)", ribbon_bull_flip & (close > ema200), ribbon_bear_flip & (close < ema200), 2.0, False, False),
            ("Ribbon Flip + SS Confluence + 200 EMA (1:2)", ribbon_bull_flip & (ss_regime == 1) & (close > ema200), ribbon_bear_flip & (ss_regime == -1) & (close < ema200), 2.0, False, False),
            ("Ribbon Flip + SS Confluence + 200 EMA (1:1.5)", ribbon_bull_flip & (ss_regime == 1) & (close > ema200), ribbon_bear_flip & (ss_regime == -1) & (close < ema200), 1.5, False, False),
            ("Ribbon Flip + 200 EMA + BreakEven at 1R (1:2)", ribbon_bull_flip & (close > ema200), ribbon_bear_flip & (close < ema200), 2.0, True, False),
            ("Pullback to ALMA + Trend Confluence + 200 EMA (1:2)",
             (ribbon_trend == 1) & (ss_regime == 1) & (close > ema200) & (low <= alma) & (close > alma) & (close > open_p),
             (ribbon_trend == -1) & (ss_regime == -1) & (close < ema200) & (high >= alma) & (close < alma) & (close < open_p),
             2.0, False, False),
            ("Pullback to ALMA + 200 EMA + BreakEven (1:2)",
             (ribbon_trend == 1) & (ss_regime == 1) & (close > ema200) & (low <= alma) & (close > alma) & (close > open_p),
             (ribbon_trend == -1) & (ss_regime == -1) & (close < ema200) & (high >= alma) & (close < alma) & (close < open_p),
             2.0, True, False),
            ("Pullback to ALMA + 200 EMA (1:1.5)",
             (ribbon_trend == 1) & (ss_regime == 1) & (close > ema200) & (low <= alma) & (close > alma) & (close > open_p),
             (ribbon_trend == -1) & (ss_regime == -1) & (close < ema200) & (high >= alma) & (close < alma) & (close < open_p),
             1.5, False, False),
        ]

        for label, buy_mask, sell_mask, rr, use_be, _ in configs:
            trades = []
            active_trade = None
            daily_losses = {}
            lookback = 12

            for i in range(250, n):
                bar_time = times.iloc[i]
                day_str = bar_time.strftime("%Y-%m-%d")
                hour = bar_time.hour
                session_ok = (Config.SESSION_START_HOUR_UTC <= hour < Config.SESSION_END_HOUR_UTC)

                # Exit logic
                if active_trade is not None:
                    tdir = active_trade["dir"]
                    sl = active_trade["sl"]
                    tp = active_trade["tp"]
                    be_hit = active_trade["be_hit"]
                    exit_res = None

                    # Check BE trigger if enabled
                    if use_be and not be_hit:
                        if tdir == "BUY" and high[i] >= active_trade["entry"] + active_trade["risk"]:
                            active_trade["sl"] = active_trade["entry"] # Trail to break-even
                            active_trade["be_hit"] = True
                            sl = active_trade["entry"]
                        elif tdir == "SELL" and low[i] <= active_trade["entry"] - active_trade["risk"]:
                            active_trade["sl"] = active_trade["entry"]
                            active_trade["be_hit"] = True
                            sl = active_trade["entry"]

                    if tdir == "BUY":
                        if low[i] <= sl:
                            exit_res = "BE" if (be_hit and sl >= active_trade["entry"]) else "LOSS"
                        elif high[i] >= tp:
                            exit_res = "WIN"
                    else:
                        if high[i] >= sl:
                            exit_res = "BE" if (be_hit and sl <= active_trade["entry"]) else "LOSS"
                        elif low[i] <= tp:
                            exit_res = "WIN"

                    if exit_res:
                        if exit_res == "WIN":
                            pnl = 25.0 * rr
                        elif exit_res == "BE":
                            pnl = 0.0
                        else:
                            pnl = -25.0
                        trades.append({"pnl": pnl, "res": exit_res, "day": active_trade["day"]})
                        if exit_res == "LOSS":
                            t_day = active_trade["day"]
                            daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                        active_trade = None

                # Entry logic
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
                            "risk": risk_dist,
                            "sl": curr_c - risk_dist,
                            "tp": curr_c + (rr * risk_dist),
                            "be_hit": False,
                            "day": day_str
                        }
                    elif sell_mask[i]:
                        recent_high = np.max(high[max(0, i - lookback):i])
                        raw_risk = recent_high - curr_c
                        risk_dist = np.clip(raw_risk, c_atr * Config.MIN_STOP_ATR, c_atr * Config.MAX_STOP_ATR)
                        active_trade = {
                            "dir": "SELL",
                            "entry": curr_c,
                            "risk": risk_dist,
                            "sl": curr_c + risk_dist,
                            "tp": curr_c - (rr * risk_dist),
                            "be_hit": False,
                            "day": day_str
                        }

            if not trades:
                print(f"{label:<55} | 0 trades")
                continue

            tdf = pd.DataFrame(trades)
            tot = len(tdf)
            wins = len(tdf[tdf["res"] == "WIN"])
            losses = len(tdf[tdf["res"] == "LOSS"])
            bes = len(tdf[tdf["res"] == "BE"])
            net = tdf["pnl"].sum()
            wr = (wins / tot) * 100
            gw = tdf[tdf["pnl"] > 0]["pnl"].sum()
            gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
            pf = (gw / gl) if gl > 0 else 999.0

            # Max Drawdown
            equity = 2000.0 + tdf["pnl"].cumsum()
            peak = equity.cummax()
            dd = peak - equity
            max_dd = dd.max()
            max_dd_pct = (max_dd / peak.max()) * 100

            print(f"{label:<55} | Trades: {tot:3d} | WR: {wr:4.1f}% ({wins}W/{losses}L/{bes}BE) | Net: ${net:+7.2f} | PF: {pf:4.2f} | MaxDD: ${max_dd:5.2f} ({max_dd_pct:3.1f}%)")

if __name__ == "__main__":
    run_tests()

