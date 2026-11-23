# Strategy 3: ICT / SMC Asian Liquidity Sweep & Judas Swing Bot

An automated Forex and Gold algorithmic trading system implementing **Smart Money Concepts (SMC)** and **Inner Circle Trader (ICT)** Asian session liquidity sweeps on the **M15 timeframe**.

Designed specifically for **Prop Firm Challenge Compliance** ($2,000 baseline capital, +8% profit target, 8% max drawdown, 5% max daily loss ceiling, and a hard 2-daily-loss circuit breaker).

---

## 🏛️ Core Mechanics

```
  00:00 - 06:45 UTC                07:00 - 11:00 UTC (London Killzone)
┌───────────────────────┐        ┌──────────────────────────────────────────────┐
│  Asian Session Range  │        │  Liquidity Raid & Market Structure Shift     │
│                       │        │                                              │
│  High: Buy-side Liq   │───────>│  Price sweeps above High -> False Breakout   │
│                       │        │  Candle rejects & closes back inside range   │
│  Low: Sell-side Liq   │───────>│  ---> SHORT ENTRY (1:1.5 to 1:2 R:R)        │
└───────────────────────┘        └──────────────────────────────────────────────┘
```

1. **Asian Session Range Tracking (00:00 – 06:45 UTC)**:
   * Dynamically tracks the Asian Session High (Buy-side Liquidity Pool) and Low (Sell-side Liquidity Pool).
   * Range filter: Only takes trades if the Asian range is between 8 and 50 pips (avoids dead markets or high-impact news spikes).

2. **London Killzone Raid Detection (07:00 – 11:00 UTC)**:
   * **Bearish Judas Swing**: Price pierces above the Asian High (triggering breakout buy orders), then closes back **below** the Asian High with a bearish rejection candle (`Close < Open`). Triggers a **SELL**.
   * **Bullish Judas Swing**: Price pierces below the Asian Low (triggering breakout sell orders), then closes back **above** the Asian Low with a bullish rejection candle (`Close > Open`). Triggers a **BUY**.

3. **Risk Management & Stop Placement**:
   * Stop Loss: Placed safely beyond the extreme sweep wick + buffer, clamped between 0.75 ATR and 2.5 ATR.
   * Take Profit: Set at a strict 1:1.5 to 1:2 Risk-to-Reward ratio.
   * Fixed Risk: $25.00 per trade (1.25% on $2,000 capital).
   * Max 2 Losses Per Day: Hard circuit breaker stops trading for the remainder of the day if 2 losses occur ($50 max loss, safely below the 5% prop firm ceiling).

---

## 🚀 Running Locally

```bash
cd strategy_3
pip install -r requirements.txt
python app.py
```

* **Dashboard**: `http://localhost:5002`
* **Health Check**: `http://localhost:5002/health`
* **Username**: `admin`
* **Password**: `oandabot2026`

---

## 📈 Running Historical Backtests

```bash
cd strategy_3
python backtest.py --symbol EURUSD --days 730 --rr 1.5
python backtest.py --symbol XAUUSD --days 730 --rr 1.5
```

---

## 🚢 Deploying to Render

1. Create a new Web Service on Render pointing to the `strategy_3/` directory (or use `render.yaml`).
2. Add your `OANDA_API_KEY` and `OANDA_ACCOUNT_ID` in the Environment Variables.
3. Configure UptimeRobot to ping `/health` every 5 minutes to prevent instance sleep.

