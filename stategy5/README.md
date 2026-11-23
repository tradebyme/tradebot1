# Strategy 5: Institutional Trend-ATR Portfolio (ITAP) Bot

Autonomous multi-asset algorithmic trading bot engineered for **Prop Firm Challenge Compliance** ($2,000, $5,000, $10,000 accounts), live order execution on **OANDA v20 REST API**, and **1-Click Cloud Deployment on Render.com**.

---

## 🏛️ Core Strategy Philosophy

Unlike failed retail scalping bots (Strategy 1 & 4 on M15) that suffered from friction decay and premature breakeven cuts, **Strategy 5** operates on macro timeframes (**H4 & H1**) using institutional trend-following principles:

1. **Macro Trend Filter**: Evaluates the **200 EMA**. Only Buys when `Close > 200 EMA`, only Sells when `Close < 200 EMA`.
2. **Momentum Entry Trigger**: Enters on candle close when the **20 EMA crosses the 50 EMA** in the trend direction.
3. **Anti-Exhaustion Proximity Filter**: Requires `|Close - 200 EMA| < 2.0x ATR(14)` to prevent entering at exhausted market tops or bottoms.
4. **Volatility Stop & Trailing Engine**:
   * **Initial Stop Loss**: Exactly `2.0x ATR(14)` transmitted directly to the broker on fill.
   * **Dynamic Peak Trailing Stop**: Ratchets at `3.0x ATR(14)` behind the highest high (longs) or lowest low (shorts) as the trade moves in profit.
5. **Multi-Asset Portfolio**:
   * **`XAU_USD` (Gold)** on **H4** (PF: 1.60, WR: 37.5%)
   * **`EUR_USD`** on **H4** (PF: 1.49, WR: 37.5%)
   * **`GBP_USD`** on **H1** (PF: 1.39, WR: 36.3%)
   * Combined 5-Year Portfolio: **Profit Factor 1.43**, **Payoff Ratio 2.48:1**, **Max Drawdown 6.88%** (safe below 10% limit).

---

## 🛡️ Prop Firm Risk Management Architecture

* **Capital Options**: `$2,000`, `$5,000`, or `$10,000` (selectable via `VIRTUAL_CAPITAL`).
* **Challenge Targets**: `+8.0%` Phase 1 target, `+5.0%` Phase 2 target.
* **Drawdown Protection**:
  * **Daily Circuit Breaker**: Trading automatically halts if daily loss touches **3.0%** (protecting the 5.0% prop firm daily limit).
  * **Daily Consecutive Loss Cap**: Halts trading after **2 losses** in a single UTC day.
  * **Overall Drawdown Floor**: Hard stops at **8.0% - 10.0%**.
* **Dual-Gear Risk Sizing**:
  * **Sprint Mode (0.75% / trade)**: $15 on $2k / $37.50 on $5k / $75 on $10k (Passes Phase 1 in ~25 trades).
  * **Funded Mode (0.50% / trade)**: $10 on $2k / $25 on $5k / $50 on $10k (4-Year Max DD drops to 6.8%, 3.8% risk of ruin).

---

## 🚀 Running Locally

```bash
# 1. Navigate to directory
cd stategy5

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start local web server
python app.py
```

* **Dashboard URL**: `http://localhost:5005`
* **Default Username**: `admin`
* **Default Password**: `oandabot2026`
* **Health Check**: `http://localhost:5005/health`

---

## 🚢 1-Click Deployment on Render.com

Deploying Strategy 5 to Render takes under 3 minutes:

### Method 1: Using `render.yaml` (Recommended Blueprint)
1. Push this folder to a GitHub repository.
2. In your [Render Dashboard](https://dashboard.render.com/), click **New** -> **Blueprint**.
3. Select your repository. Render will automatically detect `render.yaml` and configure:
   * Build Command: `pip install -r requirements.txt`
   * Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   * Health Check: `/health`
4. Enter your `OANDA_API_KEY` and `OANDA_ACCOUNT_ID` in the environment prompts.
5. Click **Apply**.

### Method 2: Manual Web Service
1. Click **New** -> **Web Service**.
2. Connect your repo and set:
   * **Root Directory**: `stategy5` (or leave empty if standalone repo)
   * **Environment**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   * **Health Check Path**: `/health`
3. In **Environment Variables**, add:
   * `OANDA_API_KEY`: `your_key_here`
   * `OANDA_ACCOUNT_ID`: `101-001-40911775-001`
   * `OANDA_ENV`: `practice` (or `live`)
   * `VIRTUAL_CAPITAL`: `2000.0`
   * `RISK_PER_TRADE_PCT`: `0.75`
   * `DASHBOARD_PASSWORD`: `oandabot2026`
4. Deploy!

### 24/7 Keep-Alive Setup (UptimeRobot)
Render free web services spin down after 15 minutes of inactivity. To keep your bot running 24/7:
1. Go to [UptimeRobot.com](https://uptimerobot.com/) (Free).
2. Create a new HTTP(s) Monitor:
   * **URL**: `https://your-bot-name.onrender.com/health`
   * **Monitoring Interval**: Every `5 minutes`.
3. Your bot will remain online 24/7 continuously scanning market bars!
