import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

def get_h1_data(symbol, days=1095):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
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
    res = np.full(len(series), np.nan)
    conv = np.convolve(series.values, weights[::-1], mode='valid')
    res[length - 1:] = conv
    return res

def run_prop_challenge_study():
    symbols = ["XAUUSD", "EURUSD", "USDJPY"]
    print("================================================================================")
    print("STRATEGY 2 PROP FIRM CHALLENGE COMPLIANCE SIMULATION (3 YEARS REAL DATA)")
    print("================================================================================")
    print("Challenge Rules: Capital $2,000 | Target: +$160 (+8%) | Max DD: -$160 (8%) | Daily Limit: -$100 (5%)")
    print("Risk per trade: $25 (1.25%) | 1:2 R:R ($50 Profit) | Max 2 Losses/Day")
    print("--------------------------------------------------------------------------------\n")

    for symbol in symbols:
        df = get_h1_data(symbol, 1095)
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

        # Confluence Entry
        buy_mask = ribbon_bull_flip & (ss_regime == 1) & (close > ema200)
        sell_mask = ribbon_bear_flip & (ss_regime == -1) & (close < ema200)

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
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": bar_time,
                        "pnl": pnl,
                        "res": exit_res,
                        "day": active_trade["day"]
                    })
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
                    active_trade = {"dir": "BUY", "entry": curr_c, "sl": curr_c - risk_dist, "tp": curr_c + (2.0 * risk_dist), "day": day_str, "entry_time": bar_time}
                elif sell_mask[i]:
                    recent_high = np.max(high[max(0, i - lookback):i])
                    risk_dist = np.clip(recent_high - curr_c, c_atr * 0.75, c_atr * 3.0)
                    active_trade = {"dir": "SELL", "entry": curr_c, "sl": curr_c + risk_dist, "tp": curr_c - (2.0 * risk_dist), "day": day_str, "entry_time": bar_time}

        trade_df = pd.DataFrame(trades)
        if len(trade_df) == 0:
            print(f"[{symbol}] No trades generated.")
            continue

        tot_trades = len(trade_df)
        wins = len(trade_df[trade_df["res"] == "WIN"])
        losses = len(trade_df[trade_df["res"] == "LOSS"])
        win_rate = (wins / tot_trades) * 100
        net_pnl = trade_df["pnl"].sum()
        gross_win = trade_df[trade_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trade_df[trade_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0

        # Running equity
        trade_df["cum_pnl"] = trade_df["pnl"].cumsum()
        trade_df["equity"] = 2000.0 + trade_df["cum_pnl"]
        peak = trade_df["equity"].cummax()
        max_dd = (peak - trade_df["equity"]).max()
        max_dd_pct = (max_dd / 2000.0) * 100

        # Rolling 30-day Challenge Simulation
        # We slide a 30-day window every 3 days across the 3 years
        start_date = trade_df["entry_time"].min()
        end_date = trade_df["exit_time"].max()
        current_start = start_date

        passed = 0
        failed_dd = 0
        failed_daily = 0
        incomplete = 0
        total_challenges = 0
        days_to_pass = []

        while current_start + timedelta(days=30) <= end_date:
            total_challenges += 1
            window_end = current_start + timedelta(days=30)
            window_trades = trade_df[(trade_df["entry_time"] >= current_start) & (trade_df["exit_time"] <= window_end)].copy()

            if len(window_trades) == 0:
                incomplete += 1
                current_start += timedelta(days=3)
                continue

            # Simulate within this window
            c_pnl = 0.0
            peak_pnl = 0.0
            daily_tracker = {}
            status = "INCOMPLETE"

            for _, tr in window_trades.iterrows():
                t_day = tr["day"]
                t_pnl = tr["pnl"]

                # Check daily loss limit ($100 max daily loss)
                daily_tracker[t_day] = daily_tracker.get(t_day, 0.0) + t_pnl
                if daily_tracker[t_day] <= -100.0:
                    failed_daily += 1
                    status = "FAILED_DAILY"
                    break

                c_pnl += t_pnl
                if c_pnl > peak_pnl:
                    peak_pnl = c_pnl

                # Check max total drawdown ($160 from starting capital or peak)
                if (peak_pnl - c_pnl) >= 160.0 or c_pnl <= -160.0:
                    failed_dd += 1
                    status = "FAILED_DD"
                    break

                # Check profit target (+$160)
                if c_pnl >= 160.0:
                    passed += 1
                    days_taken = (tr["exit_time"] - current_start).days
                    days_to_pass.append(days_taken)
                    status = "PASSED"
                    break

            if status == "INCOMPLETE":
                incomplete += 1

            current_start += timedelta(days=3)

        pass_rate = (passed / total_challenges) * 100 if total_challenges > 0 else 0
        fail_rate = ((failed_dd + failed_daily) / total_challenges) * 100 if total_challenges > 0 else 0
        avg_days = np.mean(days_to_pass) if days_to_pass else 0

        print(f"SYMBOL: {symbol} (H1 Timeframe, 3-Year Historical Study)")
        print(f"Total Trades: {tot_trades} | Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
        print(f"Net PnL: ${net_pnl:+,.2f} | Profit Factor: {profit_factor:.2f} | Max Drawdown: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
        print(f"Rolling 30-Day Challenges Simulated: {total_challenges}")
        print(f"  Passed (+8% Target Hit):       {passed:3d} ({pass_rate:4.1f}%)")
        print(f"  Failed (Max DD / Daily Breached): {failed_dd + failed_daily:3d} ({fail_rate:4.1f}%) [DD Fail: {failed_dd}, Daily Fail: {failed_daily}]")
        print(f"  Incomplete (30-day Expired):   {incomplete:3d} ({(incomplete/total_challenges)*100:4.1f}%)")
        print(f"  Average Days to Pass:          {avg_days:.1f} days\n")

if __name__ == "__main__":
    run_prop_challenge_study()

