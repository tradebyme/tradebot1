import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comprehensive_study import fetch_mt5_data
import pandas as pd
import numpy as np

def run_bench(sym: str, days: int, name: str):
    df = fetch_mt5_data(sym, days)
    highs, lows, opens, closes = df['high'].values, df['low'].values, df['open'].values, df['close'].values
    atr, rsi = df['atr_14'].values, df['rsi_14'].values
    e20, e50, e100 = df['ema_20'].values, df['ema_50'].values, df['ema_100'].values
    h1_b, h1_s = df['h1_bullish'].values, df['h1_bearish'].values
    times = df['time'].dt.to_pydatetime()

    def sim(use_h1, sl_atr, tp_atr, rr, risk):
        trades = []
        active = None
        for i in range(100, len(df)):
            b_hour = times[i].hour
            weekday = times[i].weekday()
            if active is not None:
                if active['dir'] == 'BUY':
                    if lows[i] <= active['sl']:
                        trades.append({'pnl': -risk, 'win': 0})
                        active = None
                    elif highs[i] >= active['tp']:
                        trades.append({'pnl': risk * rr, 'win': 1})
                        active = None
                else:
                    if highs[i] >= active['sl']:
                        trades.append({'pnl': -risk, 'win': 0})
                        active = None
                    elif lows[i] <= active['tp']:
                        trades.append({'pnl': risk * rr, 'win': 1})
                        active = None
            if active is not None:
                continue
            if not (7 <= b_hour < 17) or weekday in (5, 6) or (weekday == 4 and b_hour >= 20):
                continue
            prev = i - 1
            if np.isnan(atr[prev]) or atr[prev] <= 0.0001:
                continue
            r_lows = lows[i-3:i]
            r_highs = highs[i-3:i]

            h1_ok_b = (h1_b[prev] == 1.0) if use_h1 else True
            if h1_ok_b and (e20[prev] > e50[prev]) and (closes[prev] > e100[prev]):
                if any(l <= (e20[prev] + 0.75 * atr[prev]) for l in r_lows):
                    if 40.0 <= rsi[prev] <= 62.0 and closes[prev] > opens[prev] and closes[prev] > e20[prev]:
                        active = {'entry': opens[i], 'dir': 'BUY', 'sl': opens[i] - sl_atr * atr[prev], 'tp': opens[i] + tp_atr * atr[prev]}
                        continue

            h1_ok_s = (h1_s[prev] == 1.0) if use_h1 else True
            if h1_ok_s and (e20[prev] < e50[prev]) and (closes[prev] < e100[prev]):
                if any(h >= (e20[prev] - 0.75 * atr[prev]) for h in r_highs):
                    if 38.0 <= rsi[prev] <= 60.0 and closes[prev] < opens[prev] and closes[prev] < e20[prev]:
                        active = {'entry': opens[i], 'dir': 'SELL', 'sl': opens[i] + sl_atr * atr[prev], 'tp': opens[i] - tp_atr * atr[prev]}
                        continue

        tdf = pd.DataFrame(trades)
        if len(tdf) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        tot = len(tdf)
        wins = tdf['win'].sum()
        wr = (wins / tot) * 100.0
        net = tdf['pnl'].sum()
        gw = tdf[tdf['pnl'] > 0]['pnl'].sum()
        gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
        pf = gw / gl if gl > 0 else 0
        tdf['cum'] = tdf['pnl'].cumsum()
        dd = (tdf['cum'].cummax() - tdf['cum']).max()
        dd_pct = (dd / 2000.0) * 100.0
        return tot, wins, wr, net, pf, dd, dd_pct

    b_tot, b_w, b_wr, b_net, b_pf, b_dd, b_ddp = sim(use_h1=False, sl_atr=1.5, tp_atr=3.0, rr=2.0, risk=25.0)
    u_tot, u_w, u_wr, u_net, u_pf, u_dd, u_ddp = sim(use_h1=True, sl_atr=1.5, tp_atr=1.8, rr=1.2, risk=15.0)

    b_pass = "FAIL" if b_ddp >= 8.0 else "PASS"
    u_pass = "FAIL" if u_ddp >= 8.0 else "PASS"

    print(f"| {sym} ({name}) | Baseline (Old) | {b_tot} | {b_w} | {b_wr:.1f}% | ${b_net:,.0f} | {b_pf:.2f} | ${b_dd:,.0f} ({b_ddp:.1f}%) | {b_pass} |")
    print(f"| {sym} ({name}) | Upgraded (New) | {u_tot} | {u_w} | {u_wr:.1f}% | ${u_net:,.0f} | {u_pf:.2f} | ${u_dd:,.0f} ({u_ddp:.1f}%) | {u_pass} |")

if __name__ == "__main__":
    print("| Period / Asset | Model | Trades | Wins | Win Rate | Net PnL | PF | Max DD (%) | 8% Challenge |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in ['GBPUSD', 'EURUSD']:
        run_bench(s, 25, 'Nov 2026')
        run_bench(s, 90, '90 Days')
        run_bench(s, 365, '1 Year')
        run_bench(s, 730, '2 Years')
        run_bench(s, 1460, '4 Years')
