import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from strategy import ConfluenceTrendStrategy

def get_rates(symbol, tf_str="H1", days=1095):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    tf = mt5.TIMEFRAME_H1 if tf_str == "H1" else mt5.TIMEFRAME_M15
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
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

def get_trades_for_symbol(symbol, tf_str="H1", days=1095):
    df = get_rates(symbol, tf_str, days)
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    times = df["time"]

    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
    alma = fast_alma(df["close"], 34, 0.85, 6.0)
    dev = df["close"].rolling(34).std().values
    atr = ConfluenceTrendStrategy.calculate_atr(df["high"], df["low"], df["close"], 14).values

    slope_len = 3
    shift_alma = np.roll(alma, slope_len)
    shift_alma[:slope_len] = np.nan
    slope_score = np.where((atr > 0) & (~np.isnan(alma)) & (~np.isnan(shift_alma)),
                           (alma - shift_alma) / atr, 0.0)

    upper_confirm = alma + dev * 0.65
    lower_confirm = alma - dev * 0.65

    bull_setup = (slope_score > 0.08) & (close > upper_confirm)
    bear_setup = (slope_score < -0.08) & (close < lower_confirm)

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

    trend_ss, slope, threshold = ConfluenceTrendStrategy.calculate_adaptive_supersmoother(
        df["close"], 14, 50, 9, 3, 10, 0.30, 20, 0.60
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
                    "symbol": symbol,
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

    return pd.DataFrame(trades)

def simulate_challenges(trades_df, horizon_days=60, title="60-Day Horizon"):
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    start_date = trades_df["entry_time"].min()
    end_date = trades_df["exit_time"].max()
    current_start = start_date

    passed = 0
    failed_dd = 0
    failed_daily = 0
    incomplete = 0
    total = 0
    days_to_pass = []

    step = 5 # Shift every 5 days
    while current_start + timedelta(days=horizon_days) <= end_date:
        total += 1
        window_end = current_start + timedelta(days=horizon_days)
        window_trades = trades_df[(trades_df["entry_time"] >= current_start) & (trades_df["exit_time"] <= window_end)]

        c_pnl = 0.0
        peak_pnl = 0.0
        daily_tracker = {}
        status = "INCOMPLETE"

        for _, tr in window_trades.iterrows():
            t_day = tr["day"]
            t_pnl = tr["pnl"]

            daily_tracker[t_day] = daily_tracker.get(t_day, 0.0) + t_pnl
            if daily_tracker[t_day] <= -100.0:
                failed_daily += 1
                status = "FAILED_DAILY"
                break

            c_pnl += t_pnl
            if c_pnl > peak_pnl:
                peak_pnl = c_pnl

            if (peak_pnl - c_pnl) >= 160.0 or c_pnl <= -160.0:
                failed_dd += 1
                status = "FAILED_DD"
                break

            if c_pnl >= 160.0:
                passed += 1
                days_taken = (tr["exit_time"] - current_start).days
                days_to_pass.append(days_taken)
                status = "PASSED"
                break

        if status == "INCOMPLETE":
            incomplete += 1

        current_start += timedelta(days=step)

    pass_rate = (passed / total) * 100 if total > 0 else 0
    fail_rate = ((failed_dd + failed_daily) / total) * 100 if total > 0 else 0
    avg_d = np.mean(days_to_pass) if days_to_pass else 0
    print(f"[{title}] Total: {total} | Passed: {passed} ({pass_rate:.1f}%) | Failed: {failed_dd+failed_daily} ({fail_rate:.1f}%) | Expired: {incomplete} ({(incomplete/total)*100:.1f}%) | Avg Days: {avg_d:.1f}")

def run():
    print("Fetching 3 years H1 data across Portfolio (XAUUSD, EURUSD, USDJPY)...")
    df_gold = get_trades_for_symbol("XAUUSD", "H1", 1095)
    df_eur = get_trades_for_symbol("EURUSD", "H1", 1095)
    df_jpy = get_trades_for_symbol("USDJPY", "H1", 1095)

    portfolio_trades = pd.concat([df_gold, df_eur, df_jpy]).sort_values("entry_time").reset_index(drop=True)
    print(f"Total Portfolio Trades (3 Years): {len(portfolio_trades)}")
    wins = len(portfolio_trades[portfolio_trades['res'] == 'WIN'])
    wr = (wins / len(portfolio_trades)) * 100
    net = portfolio_trades['pnl'].sum()
    gw = portfolio_trades[portfolio_trades['pnl'] > 0]['pnl'].sum()
    gl = abs(portfolio_trades[portfolio_trades['pnl'] < 0]['pnl'].sum())
    pf = gw / gl if gl > 0 else 999.0
    print(f"Win Rate: {wr:.1f}% ({wins}W / {len(portfolio_trades)-wins}L) | Net PnL: ${net:+,.2f} | Profit Factor: {pf:.2f}\n")

    print("--- INDIVIDUAL ASSET CHALLENGE HORIZONS (XAUUSD GOLD) ---")
    simulate_challenges(df_gold, 30, "XAUUSD 30-Day")
    simulate_challenges(df_gold, 60, "XAUUSD 60-Day")
    simulate_challenges(df_gold, 90, "XAUUSD 90-Day")
    simulate_challenges(df_gold, 180, "XAUUSD 180-Day")

    print("\n--- MULTI-ASSET PORTFOLIO CHALLENGE HORIZONS (GOLD + EURUSD + USDJPY) ---")
    simulate_challenges(portfolio_trades, 30, "Portfolio 30-Day")
    simulate_challenges(portfolio_trades, 60, "Portfolio 60-Day")
    simulate_challenges(portfolio_trades, 90, "Portfolio 90-Day")
    simulate_challenges(portfolio_trades, 180, "Portfolio 180-Day")

if __name__ == "__main__":
    run()

