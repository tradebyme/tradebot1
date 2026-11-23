import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path("e:/OMEN/yaswanth/oanda_not/strategy_3").resolve()))
sys.path.insert(0, str(Path("e:/OMEN/yaswanth/oanda_not/ststegy2").resolve()))
sys.path.insert(0, str(Path("e:/OMEN/yaswanth/oanda_not/stategy_1").resolve()))
from run_multi_year_study import fetch_mt5_data, simulate_smc_sweep

def evaluate_challenge_lifecycle(trades_df, target_p1=160.0, target_p2=100.0, max_dd=160.0, daily_max=100.0):
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    n_trades = len(trades_df)
    
    total_evals = 0
    p1_passes = 0
    p1_fails = 0
    p1_days_list = []
    
    p2_passes = 0
    p2_fails = 0
    p2_days_list = []
    
    full_passes = 0
    
    step = 5 # Slide every 5 trades
    for i in range(0, n_trades - 40, step):
        total_evals += 1
        idx = i
        
        # --- PHASE 1 ---
        pnl = 0.0
        peak = 0.0
        daily = {}
        p1_ok = False
        start_time = trades_df.iloc[idx]["entry_time"]
        
        while idx < n_trades:
            t = trades_df.iloc[idx]
            day = t["day"]
            t_pnl = t["pnl"]
            idx += 1
            
            daily[day] = daily.get(day, 0.0) + t_pnl
            if daily[day] <= -daily_max:
                break
                
            pnl += t_pnl
            if pnl > peak:
                peak = pnl
            if (peak - pnl) >= max_dd or pnl <= -max_dd:
                break
            if pnl >= target_p1:
                p1_ok = True
                p1_days_list.append((t["exit_time"] - start_time).days)
                break
                
        if not p1_ok:
            p1_fails += 1
            continue
            
        p1_passes += 1
        
        # --- PHASE 2 ---
        pnl2 = 0.0
        peak2 = 0.0
        daily2 = {}
        p2_ok = False
        start_time2 = trades_df.iloc[idx]["entry_time"] if idx < n_trades else start_time
        
        while idx < n_trades:
            t = trades_df.iloc[idx]
            day = t["day"]
            t_pnl = t["pnl"]
            idx += 1
            
            daily2[day] = daily2.get(day, 0.0) + t_pnl
            if daily2[day] <= -daily_max:
                break
                
            pnl2 += t_pnl
            if pnl2 > peak2:
                peak2 = pnl2
            if (peak2 - pnl2) >= max_dd or pnl2 <= -max_dd:
                break
            if pnl2 >= target_p2:
                p2_ok = True
                p2_days_list.append((t["exit_time"] - start_time2).days)
                break
                
        if not p2_ok:
            p2_fails += 1
            continue
            
        p2_passes += 1
        full_passes += 1

    return {
        "total_evals": total_evals,
        "p1_passes": p1_passes,
        "p1_fails": p1_fails,
        "p1_pass_pct": (p1_passes / total_evals) * 100 if total_evals > 0 else 0,
        "p1_avg_days": np.mean(p1_days_list) if p1_days_list else 0,
        "p2_passes": p2_passes,
        "p2_fails": p2_fails,
        "p2_pass_pct": (p2_passes / p1_passes) * 100 if p1_passes > 0 else 0,
        "p2_avg_days": np.mean(p2_days_list) if p2_days_list else 0,
        "full_pass_pct": (full_passes / total_evals) * 100 if total_evals > 0 else 0
    }

def main():
    print("==========================================================================================")
    print("EXACT PROP FIRM CHALLENGE PASS/FAIL METRICS (3 YEARS REAL TICK DATA)")
    print("Challenge Targets: Phase 1 = +$160 (+8%) | Phase 2 = +$100 (+5%) | Max DD = -$160 (-8%)")
    print("==========================================================================================\n")

    # 1. Strategy 3 on EURUSD
    print(">>> 1. STRATEGY 3: ICT / SMC LIQUIDITY SWEEP (EURUSD M15) <<<")
    df_eur = fetch_mt5_data("EURUSD", 1095)
    tr_s3 = simulate_smc_sweep(df_eur, "EURUSD", rr_ratio=2.0, risk_usd=25.0)
    m_s3 = evaluate_challenge_lifecycle(tr_s3)
    print(f"Attempts Tested:  {m_s3['total_evals']}")
    print(f"Phase 1 (+8%):    Passed {m_s3['p1_passes']} times ({m_s3['p1_pass_pct']:.1f}%) | Failed {m_s3['p1_fails']} times | Avg Days to Pass: {m_s3['p1_avg_days']:.1f} days")
    print(f"Phase 2 (+5%):    Passed {m_s3['p2_passes']} times ({m_s3['p2_pass_pct']:.1f}%) | Failed {m_s3['p2_fails']} times | Avg Days to Pass: {m_s3['p2_avg_days']:.1f} days")
    print(f"Full Funded Rate: {m_s3['p2_passes']} / {m_s3['total_evals']} ({m_s3['full_pass_pct']:.1f}%)\n")

    # 2. Strategy 2 on Gold H1
    print(">>> 2. STRATEGY 2: ALMA + SUPERSMOOTHER (XAUUSD GOLD H1) <<<")
    from portfolio_horizons import get_trades_for_symbol
    tr_s2 = get_trades_for_symbol("XAUUSD", "H1", 1095)
    m_s2 = evaluate_challenge_lifecycle(tr_s2)
    print(f"Attempts Tested:  {m_s2['total_evals']}")
    print(f"Phase 1 (+8%):    Passed {m_s2['p1_passes']} times ({m_s2['p1_pass_pct']:.1f}%) | Failed {m_s2['p1_fails']} times | Avg Days to Pass: {m_s2['p1_avg_days']:.1f} days")
    print(f"Phase 2 (+5%):    Passed {m_s2['p2_passes']} times ({m_s2['p2_pass_pct']:.1f}%) | Failed {m_s2['p2_fails']} times | Avg Days to Pass: {m_s2['p2_avg_days']:.1f} days")
    print(f"Full Funded Rate: {m_s2['p2_passes']} / {m_s2['total_evals']} ({m_s2['full_pass_pct']:.1f}%)\n")

    # 3. Strategy 1 on GBPUSD M15
    print(">>> 3. STRATEGY 1: EMA TREND PULLBACK (GBPUSD M15) <<<")
    from compare_head_to_head import simulate_strategy_1, get_historical_candles
    df_gbp = get_historical_candles("GBPUSD", "M15", 730)
    tr_s1 = simulate_strategy_1(df_gbp, "GBPUSD")
    m_s1 = evaluate_challenge_lifecycle(tr_s1)
    print(f"Attempts Tested:  {m_s1['total_evals']}")
    print(f"Phase 1 (+8%):    Passed {m_s1['p1_passes']} times ({m_s1['p1_pass_pct']:.1f}%) | Failed {m_s1['p1_fails']} times | Avg Days to Pass: {m_s1['p1_avg_days']:.1f} days")
    print(f"Phase 2 (+5%):    Passed {m_s1['p2_passes']} times ({m_s1['p2_pass_pct']:.1f}%) | Failed {m_s1['p2_fails']} times | Avg Days to Pass: {m_s1['p2_avg_days']:.1f} days")
    print(f"Full Funded Rate: {m_s1['p2_passes']} / {m_s1['total_evals']} ({m_s1['full_pass_pct']:.1f}%)\n")

if __name__ == "__main__":
    main()
