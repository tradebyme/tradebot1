import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import Backtester
import pandas as pd

def main():
    symbols = ["EURUSD", "GBPUSD"]
    
    print("=" * 65)
    print("   FOREX BOT 3-DAY CHALLENGE BACKTEST (REAL MT5 DATA)   ")
    print("   Starting Capital: $2,000 | Target: +8% ($160)        ")
    print("   Daily Loss Limit: -5% ($100) | Circuit Stop: -$70    ")
    print("   Max Drawdown Limit: -8% ($160) | 1:2 Risk to Reward  ")
    print("=" * 65)
    print()

    for sym in symbols:
        print(f"--- Running Backtest on {sym} (5,000 M15 Candles) ---")
        bt = Backtester(symbol=sym, num_bars=5000)
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

        print(f"Date Range: {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')} (~2.5 months)")
        print(f"Total Trades Taken: {total_trades}")
        print(f"Winning Trades: {wins} (+$50 each)")
        print(f"Losing Trades:  {losses} (-$25 each)")
        print(f"Overall Strategy Win Rate: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Cumulative Net PnL: ${net_pnl:+.2f}")
        print()

        # 3-Day Rolling Window Evaluation
        df["date"] = df["time"].dt.date
        unique_days = sorted(list(df["date"].unique()))

        eval_3d = bt.evaluate_3day_windows(trades_df, unique_days)
        print(f">> 3-DAY CHALLENGE EVALUATION RESULTS ({eval_3d['total_windows']} Windows Tested):")
        print(f"  [PASSED] Target Hit (+8% / +$160):       {eval_3d['target_passed_count']:>2} times ({eval_3d['target_passed_pct']}%)")
        print(f"  [FAILED] Daily Loss Hit (-5% / -$100):    {eval_3d['daily_loss_breached_count']:>2} times ({eval_3d['daily_loss_breached_pct']}%)")
        print(f"  [FAILED] Max Drawdown Hit (-8% / -$160):  {eval_3d['drawdown_breached_count']:>2} times ({eval_3d['drawdown_breached_pct']}%)")
        print(f"  [NEUTRAL] Incomplete (+/- within limits): {eval_3d['neutral_count']:>2} times ({eval_3d['neutral_pct']}%)")
        print("-" * 65)
        print()

    # Combined Multi-Pair Portfolio Test
    print("=" * 65)
    print("   COMBINED PORTFOLIO TEST (EURUSD + GBPUSD SIMULTANEOUSLY)   ")
    print("=" * 65)
    
    bt_eur = Backtester("EURUSD", 5000)
    bt_gbp = Backtester("GBPUSD", 5000)
    df_eur = bt_eur.fetch_data()
    df_gbp = bt_gbp.fetch_data()
    t_eur = bt_eur.run_full_simulation(df_eur)
    t_gbp = bt_gbp.run_full_simulation(df_gbp)
    t_eur["sym"] = "EURUSD"
    t_gbp["sym"] = "GBPUSD"
    
    comb_trades = pd.concat([t_eur, t_gbp]).sort_values("entry_time").reset_index(drop=True)
    c_wins = len(comb_trades[comb_trades["result"] == "WIN"])
    c_losses = len(comb_trades[comb_trades["result"] == "LOSS"])
    c_pnl = comb_trades["pnl"].sum()
    
    print(f"Total Combined Trades: {len(comb_trades)}")
    print(f"Wins: {c_wins} | Losses: {c_losses} | Win Rate: {(c_wins/len(comb_trades))*100:.1f}%")
    print(f"Total Cumulative Profit: ${c_pnl:+.2f}")
    
    df_eur["date"] = df_eur["time"].dt.date
    u_days = sorted(list(df_eur["date"].unique()))
    c_eval = bt_eur.evaluate_3day_windows(comb_trades, u_days)
    
    print(f">> COMBINED 3-DAY CHALLENGE EVALUATION ({c_eval['total_windows']} Windows Tested):")
    print(f"  [PASSED] Target Hit (+8% / +$160):       {c_eval['target_passed_count']:>2} times ({c_eval['target_passed_pct']}%)")
    print(f"  [FAILED] Daily Loss Hit (-5% / -$100):    {c_eval['daily_loss_breached_count']:>2} times ({c_eval['daily_loss_breached_pct']}%)")
    print(f"  [FAILED] Max Drawdown Hit (-8% / -$160):  {c_eval['drawdown_breached_count']:>2} times ({c_eval['drawdown_breached_pct']}%)")
    print(f"  [NEUTRAL] Incomplete (+/- within limits): {c_eval['neutral_count']:>2} times ({c_eval['neutral_pct']}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()
