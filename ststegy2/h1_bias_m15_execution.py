import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

def get_multi_tf(symbol, days=730):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)

    m15_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    h1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
    mt5.shutdown()

    df_m15 = pd.DataFrame(m15_rates)
    df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")

    df_h1 = pd.DataFrame(h1_rates)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")

    return df_m15, df_h1

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

def test_h1_bias_m15(symbol="XAUUSD", days=730):
    df_m15, df_h1 = get_multi_tf(symbol, days)

    # Calculate H1 Bias: ALMA + SuperSmoother
    h1_alma = fast_alma(df_h1["close"], 34, 0.85, 6.0)
    h1_dev = df_h1["close"].rolling(34).std().values
    h1_atr = ConfluenceTrendStrategy.calculate_atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).values

    h1_slope_score = np.where((h1_atr > 0) & (~np.isnan(h1_alma)), (h1_alma - np.roll(h1_alma, 3)) / h1_atr, 0.0)
    h1_bull = (h1_slope_score > 0.08) & (df_h1["close"].values > h1_alma)
    h1_bear = (h1_slope_score < -0.08) & (df_h1["close"].values < h1_alma)

    df_h1["h1_trend"] = 0
    cur = 0
    for i in range(len(df_h1)):
        if h1_bull[i]:
            cur = 1
        elif h1_bear[i]:
            cur = -1
        df_h1.loc[i, "h1_trend"] = cur

    # Merge H1 trend forward into M15 candles
    df_m15 = pd.merge_asof(df_m15.sort_values("time"), df_h1[["time", "h1_trend"]].sort_values("time"),
                           on="time", direction="backward")

    # M15 indicators
    m15_alma = fast_alma(df_m15["close"], Config.ALMA_LENGTH, Config.ALMA_OFFSET, Config.ALMA_SIGMA)
    m15_dev = df_m15["close"].rolling(window=Config.DEV_LENGTH).std().values
    m15_atr = ConfluenceTrendStrategy.calculate_atr(df_m15["high"], df_m15["low"], df_m15["close"], Config.ATR_LENGTH).values

    m15_upper = m15_alma + m15_dev * Config.DEV_MULT
    m15_lower = m15_alma - m15_dev * Config.DEV_MULT

    close = df_m15["close"].values
    high = df_m15["high"].values
    low = df_m15["low"].values
    times = df_m15["time"]
    h1_trend = df_m15["h1_trend"].values

    # M15 Pullback to ALMA bounce when aligned with H1 Trend
    m15_buy = (h1_trend == 1) & (low <= m15_alma) & (close > m15_alma) & (close > df_m15["open"].values)
    m15_sell = (h1_trend == -1) & (high >= m15_alma) & (close < m15_alma) & (close < df_m15["open"].values)

    trades = []
    active_trade = None
    daily_losses = {}
    lookback = 12

    for i in range(250, len(df_m15)):
        bar_time = times.iloc[i]
        day_str = bar_time.strftime("%Y-%m-%d")
        curr_c = close[i]
        c_atr = m15_atr[i]

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

            if m15_buy[i]:
                recent_low = np.min(low[max(0, i - lookback):i])
                risk_dist = np.clip(curr_c - recent_low, c_atr * 0.75, c_atr * 2.5)
                active_trade = {"dir": "BUY", "entry": curr_c, "sl": curr_c - risk_dist, "tp": curr_c + (2.0 * risk_dist), "day": day_str}
            elif m15_sell[i]:
                recent_high = np.max(high[max(0, i - lookback):i])
                risk_dist = np.clip(recent_high - curr_c, c_atr * 0.75, c_atr * 2.5)
                active_trade = {"dir": "SELL", "entry": curr_c, "sl": curr_c + risk_dist, "tp": curr_c - (2.0 * risk_dist), "day": day_str}

    if not trades:
        print(f"{symbol} H1-Bias M15 Pullback: 0 trades")
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

    print(f"{symbol} (H1-Bias + M15 Pullback) | Trades: {tot} | WR: {wr:.1f}% ({wins}W/{losses}L) | Net: ${net:+,.2f} | PF: {pf:.2f} | MaxDD: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

def main():
    print("Multi-Timeframe Test (H1 Trend Bias + M15 ALMA Pullback Execution):")
    for s in ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD"]:
        test_h1_bias_m15(s, 730)

if __name__ == "__main__":
    main()
