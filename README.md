# OANDA Prop Firm Multi-Bot Cloud Gateway

Autonomous algorithmic trading suite engineered for **Prop Firm Challenge Compliance** ($2,000 / $5,000 / $10,000 / $100,000 accounts), live order execution on **OANDA v20 REST API**, and **1-Click Cloud Deployment on Render.com**.

---

## 🚀 Unified Multi-Bot Navigation Architecture

This repository hosts isolated, production-grade trading bots accessible through a single, unified web gateway:

* **Page 1: Strategy 1 (M15 Scalper)**:
  * Route: `/` or `/strategy1`
  * Strategy: London/NY M15 momentum pullback scalper with 1:2 R:R.
* **Page 2: APEX SNIPER / Strategy 5 (Institutional Trend-ATR Portfolio)**:
  * Route: `/apex` or `/strategy5`
  * Strategy: 5-Year MetaTrader 5 Verified (+41.3% net profit, PF 1.44, 2.49x payoff, 0 daily limit breaches).
  * Macro Trend Filter (200 EMA) + Momentum Crossover (20/50 EMA) + Anti-Exhaustion Guard (< 2.0x ATR) + Dynamic Peak-ATR (3.0x ATR) Trailing Stop.
  * Multi-Asset Portfolio: `XAU_USD` (H4), `EUR_USD` (H4), `GBP_USD` (H1).

Both bots feature a sleek, glassmorphic top navigation bar enabling instant 1-click toggling between Bot 1 and Bot 2 without leaving your browser tab.

---

## 🛡️ Isolated Process & Credential Architecture

Both bots execute inside completely isolated background subprocesses with dedicated risk managers, state files, and broker order feeds:

* **Separate API Keys & Accounts**:
  * Set `OANDA_API_KEY_1` and `OANDA_ACCOUNT_ID_1` for Strategy 1.
  * Set `OANDA_API_KEY_APEX` and `OANDA_ACCOUNT_ID_APEX` for APEX SNIPER.
  * If using a single shared demo account, set `OANDA_API_KEY` and `OANDA_ACCOUNT_ID`; each bot independently isolates trades using custom client transaction tags and internal virtual capital trackers.
* **Proactive Circuit Breakers**:
  * Daily Loss Limit: 3.0% - 3.5% proactive freeze (safely protecting the 5.0% prop firm max daily limit).
  * Maximum Drawdown Floor: 8.0% hard halt.

---

## ☁️ Deployment on Render.com (Single Web Service)

1. Connect your GitHub repository `https://github.com/tradebyme/tradebot1.git` to Render.
2. Select **Web Service** using `Python 3.11`.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python app.py`
5. Configure your Environment Variables (`OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, etc.).
6. Click **Deploy**! Both Page 1 (`/strategy1`) and Page 2 (`/apex`) will be live on your free Render URL.
