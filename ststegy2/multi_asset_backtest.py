import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

def get_rates(symbol, tf_str="M15", days=730):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    tf = mt5.TIMEFRAME_M15 if tf_str == "M15" else mt5.TIMEFRAME_H1
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

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

def evaluate_symbol(symbol, tf_str="M15", days=730):
    df = get_rates(symbol, tf_str, days)
    if df is None:
        print(f"Failed to fetch data for {symbol}")
        return

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
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

    # Setup: Trend Confluence + 200 EMA Filter
    buy_mask = ribbon_bull_flip & (ss_regime == 1) & (close > ema200)
    sell_mask = ribbon_bear_flip & (ss_regime == -1) & (close < ema200)

    # Simulation with $25 risk, 1:2 R:R ($50 win), Max 2 daily losses
    trades = []
    active_trade = None
    daily_losses = {}
    lookback = 12

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
                trades.append({"pnl": pnl, "res": exit_res, "day": active_trade["day"]})
                if exit_res == "LOSS":
                    t_day = active_trade["day"]
                    daily_losses[t_day] = daily_losses.get(t_day, 0) + 1
                active_trade = None

        if active_trade is None:
            if daily_losses.get(day_str, 0) >= 2:
                continue
            if np.isnan(c_atr) or c_atr <= 0:
                continue
            if buy_mask[i]:
                recent_low = np.min(low[max(0, i - lookback):i])
                risk_dist = np.clip(curr_c - recent_low, c_atr * 0.75, c_atr * 3.0)
                active_trade = {"dir": "BUY", "entry": curr_c, "sl": curr_c - risk_dist, "tp": curr_c + (2.0 * risk_dist), "day": day_str}
            elif sell_mask[i]:
                recent_high = np.max(high[max(0, i - lookback):i])
                risk_dist = np.clip(recent_high - curr_c, c_atr * 0.75, c_atr * 3.0)
                active_trade = {"dir": "SELL", "entry": curr_c, "sl": curr_c + risk_dist, "tp": curr_c - (2.0 * risk_dist), "day": day_str}

    if not trades:
        print(f"{symbol:<8} ({tf_str}) | 0 trades")
        return

    tdf = pd.DataFrame(trades)
    tot = len(tdf)
    wins = len(tdf[tdf["res"] == "WIN"])
    losses = len(tdf[tdf["res"] == "LOSS"])
    net = tdf["pnl"].sum()
    wr = (wins / tot) * 100
    gw = tdf[tdf["pnl"] > 0]["pnl"].sum()
    gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
    pf = (gw / gl) if gl > 0 else 999.0

    equity = 2000.0 + tdf["pnl"].cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd = dd.max()
    max_dd_pct = (max_dd / peak.max()) * 100

    print(f"{symbol:<8} ({tf_str:<3}) | Bars: {n:6d} | Trades: {tot:3d} | WR: {wr:4.1f}% ({wins}W/{losses}L) | Net: ${net:+7.2f} | PF: {pf:4.2f} | MaxDD: ${max_dd:5.2f} ({max_dd_pct:4.1f}%)")

def main():
    print("Testing M15 Timeframe (730 Days):")
    for s in ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD"]:
        evaluate_symbol(s, "M15", 730)

    print("\nTesting H1 Timeframe (1095 Days / 3 Years):")
    for s in ["GBPUSD", "EURUSD", "USDJPY", "XAUUSD"]:
        evaluate_symbol(s, "H1", 1095)

if __name__ == "__main__":
    main()
