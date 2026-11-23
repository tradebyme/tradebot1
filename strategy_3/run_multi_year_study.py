import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiYearStudy")

def fetch_mt5_data(symbol: str, days: int = 1095) -> pd.DataFrame:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def simulate_smc_sweep(df: pd.DataFrame, symbol: str, rr_ratio: float = 1.5, risk_usd: float = 25.0) -> pd.DataFrame:
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    days = df["date"].unique()

    is_gold = ("XAU" in symbol or "GOLD" in symbol)
    is_jpy = ("JPY" in symbol)
    pip_scale = 0.10 if is_gold else (0.01 if is_jpy else 0.0001)
    sweep_buffer = (5.0 * pip_scale) if is_gold else (Config.SWEEP_BUFFER_PIPS * pip_scale)

    min_range = 30.0 if is_gold else (15.0 if is_jpy else 8.0)
    max_range = 350.0 if is_gold else (90.0 if is_jpy else 55.0)

    trades = []
    daily_losses = {}
    max_daily_losses = 2

    for d in days:
        day_str = str(d)
        if daily_losses.get(day_str, 0) >= max_daily_losses:
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

            # Bearish Sweep: Swept Asian High, closed back below Asian High, Red candle
            if h > asian_high and c < asian_high and c < o:
                entry = c
                sl = h + sweep_buffer
                raw_risk = sl - entry
                if raw_risk <= 0:
                    continue
                tp = entry - (rr_ratio * raw_risk)
                entered = True
                direction = "SELL"
                start_i = day_df.index[day_df["time"] == bar_time][0]

            # Bullish Sweep: Swept Asian Low, closed back above Asian Low, Green candle
            elif l < asian_low and c > asian_low and c > o:
                entry = c
                sl = l - sweep_buffer
                raw_risk = entry - sl
                if raw_risk <= 0:
                    continue
                tp = entry + (rr_ratio * raw_risk)
                entered = True
                direction = "BUY"
                start_i = day_df.index[day_df["time"] == bar_time][0]

            if entered:
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

                if result is None:
                    exit_time = day_df["time"].iloc[-1]
                    last_c = day_df["close"].iloc[-1]
                    if direction == "BUY":
                        pnl = ((last_c - entry) / raw_risk) * risk_usd
                    else:
                        pnl = ((entry - last_c) / raw_risk) * risk_usd
                    result = "WIN" if pnl > 0 else "LOSS"
                else:
                    pnl = (risk_usd * rr_ratio) if result == "WIN" else -risk_usd

                trades.append({
                    "symbol": symbol,
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

def simulate_prop_challenges(trade_df: pd.DataFrame, target_pnl: float = 160.0, max_dd_limit: float = 160.0):
    if len(trade_df) == 0:
        return 0, 0, 0, 0.0

    trade_df = trade_df.sort_values("entry_time").reset_index(drop=True)
    start_date = trade_df["entry_time"].min()
    end_date = trade_df["exit_time"].max()
    current_start = start_date

    passed = 0
    failed = 0
    incomplete = 0
    total = 0
    days_to_pass = []

    while current_start + timedelta(days=60) <= end_date:
        total += 1
        window_end = current_start + timedelta(days=60)
        w_trades = trade_df[(trade_df["entry_time"] >= current_start) & (trade_df["exit_time"] <= window_end)]

        c_pnl = 0.0
        peak_pnl = 0.0
        daily_tracker = {}
        status = "INCOMPLETE"

        for _, tr in w_trades.iterrows():
            t_day = tr["day"]
            t_pnl = tr["pnl"]

            daily_tracker[t_day] = daily_tracker.get(t_day, 0.0) + t_pnl
            if daily_tracker[t_day] <= -100.0:
                failed += 1
                status = "FAILED"
                break

            c_pnl += t_pnl
            if c_pnl > peak_pnl:
                peak_pnl = c_pnl

            if (peak_pnl - c_pnl) >= max_dd_limit or c_pnl <= -max_dd_limit:
                failed += 1
                status = "FAILED"
                break

            if c_pnl >= target_pnl:
                passed += 1
                days_taken = (tr["exit_time"] - current_start).days
                days_to_pass.append(days_taken)
                status = "PASSED"
                break

        if status == "INCOMPLETE":
            incomplete += 1

        current_start += timedelta(days=5)

    pass_rate = (passed / total) * 100 if total > 0 else 0
    avg_days = np.mean(days_to_pass) if days_to_pass else 0
    return total, passed, pass_rate, avg_days

def run_study():
    days = 1095 # 3 Full Years
    symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"]

    print("==========================================================================================")
    print(f"STRATEGY 3: 3-YEAR COMPREHENSIVE HISTORICAL STUDY ({days} DAYS / ~75,000 M15 BARS)")
    print("Baseline: $2,000 Capital | $25 Fixed Risk | Max 2 Daily Losses | London Killzone Sweeps")
    print("==========================================================================================\n")

    all_portfolio_trades_rr15 = []
    all_portfolio_trades_rr20 = []

    for sym in symbols:
        df = fetch_mt5_data(sym, days)
        if df is None:
            print(f"Failed to fetch {sym}.")
            continue

        n_bars = len(df)

        # Test R:R = 1:1.5
        trades_15 = simulate_smc_sweep(df, sym, rr_ratio=1.5, risk_usd=25.0)
        # Test R:R = 1:2.0
        trades_20 = simulate_smc_sweep(df, sym, rr_ratio=2.0, risk_usd=25.0)

        all_portfolio_trades_rr15.append(trades_15)
        all_portfolio_trades_rr20.append(trades_20)

        def print_metrics(tdf, rr):
            tot = len(tdf)
            if tot == 0:
                print(f"  [1:{rr:.1f} R:R] 0 trades.")
                return
            wins = len(tdf[tdf["result"] == "WIN"])
            losses = len(tdf[tdf["result"] == "LOSS"])
            wr = (wins / tot) * 100
            net = tdf["pnl"].sum()
            gw = tdf[tdf["pnl"] > 0]["pnl"].sum()
            gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
            pf = (gw / gl) if gl > 0 else 999.0

            tdf["cum_pnl"] = tdf["pnl"].cumsum()
            tdf["equity"] = 2000.0 + tdf["cum_pnl"]
            peak = tdf["equity"].cummax()
            max_dd = (peak - tdf["equity"]).max()
            max_dd_pct = (max_dd / 2000.0) * 100

            print(f"  [R:R 1:{rr:.1f}] Trades: {tot:4d} | Win Rate: {wr:4.1f}% ({wins}W / {losses}L) | Net PnL: ${net:+8.2f} ({(net/2000.0)*100:+.1f}%) | PF: {pf:4.2f} | MaxDD: ${max_dd:6.2f} ({max_dd_pct:4.1f}%)")

        print(f">>> {sym} ({n_bars:,} M15 Candles / 3 Years) <<<")
        print_metrics(trades_15, 1.5)
        print_metrics(trades_20, 2.0)
        print()

    # Portfolio Analysis (EURUSD + GBPUSD + XAUUSD)
    port_15 = pd.concat(all_portfolio_trades_rr15).sort_values("entry_time").reset_index(drop=True)
    port_20 = pd.concat(all_portfolio_trades_rr20).sort_values("entry_time").reset_index(drop=True)

    print("==========================================================================================")
    print("MULTI-ASSET PORTFOLIO SIMULATION (EURUSD + GBPUSD + XAUUSD + USDJPY)")
    print("==========================================================================================")
    print("--- PORTFOLIO AT 1:1.5 RISK-TO-REWARD ---")
    tot_15 = len(port_15)
    wins_15 = len(port_15[port_15["result"] == "WIN"])
    net_15 = port_15["pnl"].sum()
    pf_15 = port_15[port_15["pnl"] > 0]["pnl"].sum() / abs(port_15[port_15["pnl"] < 0]["pnl"].sum())
    print(f"Total Trades: {tot_15} | Win Rate: {(wins_15/tot_15)*100:.1f}% | Net PnL: ${net_15:+,.2f} | PF: {pf_15:.2f}")
    tot_ch, pass_ch, pass_rt, avg_d = simulate_prop_challenges(port_15)
    print(f"Rolling 60-Day Prop Challenges: {tot_ch} Simulated | Passed (+8%): {pass_ch} ({pass_rt:.1f}%) | Avg Days: {avg_d:.1f}\n")

    print("--- PORTFOLIO AT 1:2.0 RISK-TO-REWARD ---")
    tot_20 = len(port_20)
    wins_20 = len(port_20[port_20["result"] == "WIN"])
    net_20 = port_20["pnl"].sum()
    pf_20 = port_20[port_20["pnl"] > 0]["pnl"].sum() / abs(port_20[port_20["pnl"] < 0]["pnl"].sum())
    print(f"Total Trades: {tot_20} | Win Rate: {(wins_20/tot_20)*100:.1f}% | Net PnL: ${net_20:+,.2f} | PF: {pf_20:.2f}")
    tot_ch2, pass_ch2, pass_rt2, avg_d2 = simulate_prop_challenges(port_20)
    print(f"Rolling 60-Day Prop Challenges: {tot_ch2} Simulated | Passed (+8%): {pass_ch2} ({pass_rt2:.1f}%) | Avg Days: {avg_d2:.1f}\n")

if __name__ == "__main__":
    run_study()

