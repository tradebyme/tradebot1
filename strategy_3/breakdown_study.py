import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multi_year_study import fetch_mt5_data, simulate_smc_sweep

def run_breakdown():
    df = fetch_mt5_data("EURUSD", 1095)
    
    # 1. Test 1:2 R:R
    trades_20 = simulate_smc_sweep(df, "EURUSD", rr_ratio=2.0, risk_usd=25.0)
    trades_20["year_month"] = pd.to_datetime(trades_20["entry_time"]).dt.strftime("%Y-%m")
    trades_20["year"] = pd.to_datetime(trades_20["entry_time"]).dt.year

    print("=== EURUSD (1:2 R:R) YEARLY BREAKDOWN ===")
    for y, y_df in trades_20.groupby("year"):
        tot = len(y_df)
        wins = len(y_df[y_df["result"] == "WIN"])
        wr = (wins / tot) * 100
        pnl = y_df["pnl"].sum()
        print(f"Year {y}: Trades={tot:3d} | WR={wr:4.1f}% ({wins}W / {tot-wins}L) | Net PnL=${pnl:+8.2f}")

    # 2. Test 1:1 R:R (For above 50% Win Rate)
    trades_10 = simulate_smc_sweep(df, "EURUSD", rr_ratio=1.0, risk_usd=25.0)
    trades_10["year_month"] = pd.to_datetime(trades_10["entry_time"]).dt.strftime("%Y-%m")
    trades_10["year"] = pd.to_datetime(trades_10["entry_time"]).dt.year

    print("\n=== EURUSD (1:1 R:R) YEARLY BREAKDOWN (HIGH WIN RATE MODEL) ===")
    for y, y_df in trades_10.groupby("year"):
        tot = len(y_df)
        wins = len(y_df[y_df["result"] == "WIN"])
        wr = (wins / tot) * 100
        pnl = y_df["pnl"].sum()
        print(f"Year {y}: Trades={tot:3d} | WR={wr:4.1f}% ({wins}W / {tot-wins}L) | Net PnL=${pnl:+8.2f}")

    # Total 1:1 metrics
    tot_10 = len(trades_10)
    wins_10 = len(trades_10[trades_10["result"] == "WIN"])
    wr_10 = (wins_10 / tot_10) * 100
    pnl_10 = trades_10["pnl"].sum()
    print(f"\nOverall 1:1 R:R: Total Trades={tot_10} | Win Rate={wr_10:.1f}% | Net PnL=${pnl_10:+,.2f}")

    # Monthly sample for 2026
    print("\n=== 2026 MONTHLY BREAKDOWN (1:1 R:R) ===")
    m_2026 = trades_10[trades_10["year"] == 2026]
    for ym, m_df in m_2026.groupby("year_month"):
        tot = len(m_df)
        wins = len(m_df[m_df["result"] == "WIN"])
        wr = (wins / tot) * 100
        pnl = m_df["pnl"].sum()
        print(f"Month {ym}: Trades={tot:2d} | WR={wr:4.1f}% ({wins}W / {tot-wins}L) | Net PnL=${pnl:+7.2f}")

if __name__ == "__main__":
    run_breakdown()

