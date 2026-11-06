# OANDA v20 Automated Forex Trading Bot & Dashboard

An institutional-grade, fully automated Forex trading bot and real-time web dashboard built with **Python (Flask)** and the **OANDA v20 REST API**. Designed specifically to pass rapid evaluation challenges (e.g. **+8% in 3 days on $2,000 capital**) with built-in mathematical risk limits and circuit breakers.

Optimized to run **100% FREE 24/5** on **Render Free Web Service** paired with **UptimeRobot**.

---

## 📊 Prop Challenge Architecture & Mathematical Rules

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Challenge Capital** | **$2,000.00** | Starting account reference equity. |
| **Target Profit** | **+8.0% (+$160.00)** | Target to achieve. Once reached, trading locks automatically to secure the pass. |
| **Max Total Drawdown** | **-8.0% (-$160.00)** | Hard floor at $1,840.00. Bot terminates if equity reaches this floor. |
| **Daily Loss Limit** | **-5.0% (-$100.00)** | Hard daily loss ceiling. |
| **Daily Circuit Breaker** | **-3.5% (-$70.00)** | Soft circuit stop: Bot stops taking trades if daily loss hits -$70 to avoid slippage breaching the hard 5% limit. |
| **Risk Per Trade** | **1.25% ($25.00)** | Dynamic lot sizing guarantees maximum loss per trade is exactly $25. |
| **Risk-to-Reward (R:R)** | **1 : 2** | Risk $25 to make $50 per trade. |
| **Target Requirement** | **4 net wins** | With a 1:2 R:R, just 4 net winning trades in 3 days achieves the +$160 goal. |

---

## 🎯 The Strategy: London & New York Momentum Pullback (M15)

1. **Market & Session Filters:**
   * Trades only during peak liquidity hours: **07:00 UTC to 17:00 UTC** (London Open & New York Overlap).
   * Automatically idles during the weekend market close (Friday 21:00 UTC to Sunday 21:00 UTC).
   * Blocks entries if bid/ask spread exceeds **2.5 pips**.
2. **Trend Confirmation:**
   * **Bullish (BUY):** 20 EMA > 50 EMA and Price > 100 EMA.
   * **Bearish (SELL):** 20 EMA < 50 EMA and Price < 100 EMA.
3. **Pullback & Trigger:**
   * Price retraces towards the 20 EMA while RSI(14) resets in the 40–60 momentum zone.
   * Enters upon a confirmed closed candle bouncing back in the trend direction.
4. **Guaranteed Risk Protection:**
   * Every market order is sent with **immediate StopLossOnFill** and **TakeProfitOnFill** attached directly in OANDA's broker engine. Even if the bot server loses internet, your stop loss is guaranteed.

---

## 🖥️ Live Web Dashboard Features

* **Live KPIs:** Real-time challenge target progress (+8% meter), Daily loss safety gauge, Challenge equity ($2k baseline), and OANDA account NAV.
* **Equity Curve Chart:** Interactive Chart.js plotting actual equity against the $2,160 Target line and $1,840 Stop line.
* **Active Positions:** Displays open trades, ticket IDs, unrealized PnL, with individual "Close" buttons.
* **Trade History:** Table of recently closed trades with realized PnL.
* **Activity Stream:** Real-time terminal log displaying indicator calculations, candle checks, and order executions.
* **Manual Controls:**
  * **Pause / Resume Bot**
  * **Force Scan Now**
  * **Emergency Close All Trades**
  * **Reset Challenge State**

---

## 🚀 Local Setup & Testing

### 1. Requirements
Ensure Python 3.9+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify `.env` Configuration
Open `.env` and verify your OANDA credentials and security password:
```env
OANDA_API_KEY=your_oanda_personal_access_token
OANDA_ACCOUNT_ID=your_oanda_account_id
OANDA_ENV=practice

# Dashboard Security Credentials
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=oandabot2026
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🌐 100% FREE 24/5 Deployment to Render

You can host this bot on Render completely free without a credit card.

### Step 1: Push Code to GitHub
1. Create a new private repository on GitHub (e.g. `oanda-trading-bot`).
2. Push the files inside `stategy_1/` to your repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy OANDA bot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/oanda-trading-bot.git
   git push -u origin main
   ```

### Step 2: Deploy as a Free Web Service on Render
1. Go to [https://render.com](https://render.com) and sign up (Free, No Credit Card required).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository.
4. Fill in the settings:
   * **Name:** `oanda-forex-bot`
   * **Region:** Any (e.g. Oregon or Frankfurt)
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   * **Instance Type:** `Free`
5. Under **Environment Variables**, add:
   * `OANDA_API_KEY`: *(Your OANDA token)*
   * `OANDA_ACCOUNT_ID`: *(Your OANDA account ID)*
   * `OANDA_ENV`: `practice`
   * `VIRTUAL_CAPITAL`: `2000.0`
6. Click **Create Web Service**. Render will build and launch your dashboard in ~2 minutes!

---

## ⏰ Step 3: Keep It Running 24/5 for Free (UptimeRobot)

Render free web services normally go to sleep after 15 minutes of inactivity. To keep your trading bot active 24 hours a day during market hours:

1. Go to [https://uptimerobot.com](https://uptimerobot.com) and create a free account (No credit card).
2. Click **Add New Monitor**:
   * **Monitor Type:** `HTTP(s)`
   * **Friendly Name:** `OANDA Bot Keep-Alive`
   * **URL:** `https://your-render-app-name.onrender.com/health`
   * **Monitoring Interval:** `Every 5 minutes`
3. Click **Create Monitor**.

UptimeRobot will ping `/health` every 5 minutes. This completely prevents Render from sleeping, keeping your bot scanning and trading 24/5 continuously at **$0.00 cost**!

---

## 🛡️ Circuit Breakers & Fail-safes Summary

* **Slippage Protection:** The soft daily stop triggers at **-$70 (-3.5%)**, leaving a $30 cushion so spreads or slippage never breach your hard 5% limit.
* **Rollover Reset:** At **00:00 UTC**, daily stats and circuit breakers automatically reset for the new trading day.
* **State Persistence:** `state.json` persists your challenge metrics across server restarts.
* **Direct Broker SL/TP:** Stop Loss and Take Profit are submitted directly to OANDA with each order. Even if your server or internet disconnects, the broker automatically manages your exits.

