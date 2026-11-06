import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import Backtester
import pandas as pd

def run_2year_study():
    symbols = ["GBPUSD", "EURUSD"]

    print("=" * 72)
    print("      2-YEAR COMPREHENSIVE FOREX BACKTEST (REAL MT5 DATA)       ")
    print("      Period: November 2024 - November 2026 (50,000 M15 Candles) ")
    print("      Capital: $2,000 | Target: +8% ($160) | Daily Stop: 5%     ")
    print("      Stop Loss: 1.5x ATR ($25 risk) | Take Profit: 3.0x ATR    ")
    print("=" * 72)
    print()

    for sym in symbols:
        print(f"================== {sym} 2-YEAR HISTORICAL STUDY ==================")
        bt = Backtester(symbol=sym, days=730)
        df = bt.fetch_data()
        trades_df = bt.run_full_simulation(df)

        total_trades = len(trades_df)
        if total_trades == 0:
            print("No trades found.")
            continue

        wins = len(trades_df[trades_df["result"] == "WIN"])
        losses = len(trades_df[trades_df["result"] == "LOSS"])
        win_rate = (wins / total_trades) * 100.0
        net_pnl = trades_df["pnl"].sum()
        profit_factor = (wins * 50.0) / (losses * 25.0) if losses > 0 else 999.0

        # Equity curve & Max Drawdown over 2 years
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
        trades_df["peak"] = trades_df["cum_pnl"].cummax()
        trades_df["drawdown"] = trades_df["peak"] - trades_df["cum_pnl"]
        max_dd = trades_df["drawdown"].max()
        max_dd_pct = (max_dd / bt.capital) * 100.0

        start_date = df["time"].iloc[0].strftime("%Y-%m-%d")
        end_date = df["time"].iloc[-1].strftime("%Y-%m-%d")

        print(f"• Historical Period:     {start_date} to {end_date} (2 Full Years)")
        print(f"• Total M15 Bars Tested: {len(df):,} bars")
        print(f"• Total Trades Taken:    {total_trades:,}")
        print(f"• Winning Trades:        {wins:,} (+${wins * 50:,.2f})")
        print(f"• Losing Trades:         {losses:,} (-${losses * 25:,.2f})")
        print(f"• Overall Win Rate:      {win_rate:.2f}%")
        print(f"• Profit Factor:         {profit_factor:.2f}")
        print(f"• Max Drawdown in 2 Yrs: ${max_dd:.2f} ({max_dd_pct:.1f}% of $2k)")
        print(f"• Total Net Profit:      ${net_pnl:+,.2f} ({(net_pnl/bt.capital)*100:+.1f}% on $2k)")
        print()

        # Multi-Duration Challenge Analysis
        df["date"] = df["time"].dt.date
        unique_days = sorted(list(df["date"].unique()))

        print(">> CHALLENGE EVALUATION ACROSS ALL ROLLING PERIODS IN 2 YEARS:")
        for days in [3, 5, 10]:
            ev = bt.evaluate_window_duration(trades_df, unique_days, window_days=days)
            print(f"  [ {days:2d}-DAY WINDOWS ] ({ev['total_windows']} Total Windows Tested)")
            print(f"    - Target Passed (+8% / +$160):      {ev['target_passed_count']:>3} times ({ev['target_passed_pct']:>5.1f}%)")
            print(f"    - Daily Loss Breached (-5% / -$100): {ev['daily_loss_breached_count']:>3} times ({ev['daily_loss_breached_pct']:>5.1f}%)")
            print(f"    - Max Drawdown Hit (-8% / -$160):   {ev['drawdown_breached_count']:>3} times ({ev['drawdown_breached_pct']:>5.1f}%)")
            print(f"    - Survived / Neutral:               {ev['neutral_count']:>3} times ({ev['neutral_pct']:>5.1f}%)")
            print()
        print("-" * 72)
        print()

if __name__ == "__main__":
    run_2year_study()

