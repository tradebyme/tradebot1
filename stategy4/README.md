# Strategy 4: Adaptive SuperSmoother Trading Bot & Dashboard

An institutional-grade, fully automated Forex trading bot and real-time web dashboard built with **Python (Flask)** and the **OANDA v20 REST API**. 

Powered by **SchizoQuant's Adaptive SuperSmoother** indicator, combining a volatility-normalized adaptive moving average, John Ehlers' 2-Pole SuperSmoother digital filter, and Kaufman's Efficiency Ratio with prop firm challenge mathematical risk safeguards.

Optimized to run **100% FREE 24/5** on **Render Free Web Service** paired with **UptimeRobot**.

---

## 🔬 Pine Script In-Depth Technical Analysis

The trading engine directly implements the mathematical formulation from [`pinescript.txt`](file:///e:/OMEN/yaswanth/oanda_not/stategy4/pinescript.txt):

```
                   ┌───────────────────────────────────────────────┐
                   │    Raw Price Input (Close, Completed Bar)     │
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 1. Volatility Position & Dynamic Period       │
                   │    s = stdev(ΔP, 9), sMean = sma(s, 9)        │
                   │    Period adapts between 14 (fast) & 50 (slow)│
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 2. Adaptive Moving Average (AMA)              │
                   │    alpha = 2.0 / (adaptiveLength + 1.0)       │
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 3. John Ehlers 2-Pole SuperSmoother Filter    │
                   │    2-pole recursive IIR filter (L=3)          │
                   │    Eliminates spectral noise & zero phase lag │
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 4. Kaufman Efficiency Ratio (ER)              │
                   │    Efficiency = Direction / Noise (10 bars)   │
                   │    FastWeight = 0.30 + 0.70 * Efficiency      │
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 5. Blended Trend Line & Slope Noise Filter    │
                   │    Trend = AMA * FastWeight + Smooth * Slow   │
                   │    Slope = Trend - Trend[1]                   │
                   │    Threshold = RMA(|Slope|, 20) * 0.60        │
                   └───────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │ 6. State Machine (SQ) Regime Transitions      │
                   │    Slope > +Thresh -> SQ = 1  (BULLISH)       │
                   │    Slope < -Thresh -> SQ = -1 (BEARISH)       │
                   │    LongSignal:  SQ == 1  and SQ[1] != 1       │
                   │    ShortSignal: SQ == -1 and SQ[1] != -1      │
                   └───────────────────────────────────────────────┘
```

### 1. Volatility-Adaptive Period
Instead of fixed-length moving averages that either lag in trends or whip in consolidation:
* Calculates the rolling standard deviation $s$ of bar-to-bar price changes ($\Delta P = Close - Close[1]$) over a 9-bar window.
* Evaluates $s$ against a dynamic corridor $[0.25 \times sMean, 1.75 \times sMean]$.
* When volatility is high ($s \ge sMax$), the period shortens to **14 bars** for rapid responsiveness.
* When volatility is low ($s \le sMin$), the period extends to **50 bars** for maximum noise dampening.

### 2. John Ehlers 2-Pole SuperSmoother Filter
* Applies John Ehlers' 2-pole recursive digital filter directly to the adaptive moving average.
* Filter coefficients derived from polar coordinates:
  $$a_1 = \exp\left(-\frac{\sqrt{2}\pi}{smoothLength}\right), \quad b_1 = 2 a_1 \cos\left(\frac{\sqrt{2}\pi}{smoothLength}\right)$$
  $$c_2 = b_1, \quad c_3 = -a_1^2, \quad c_1 = 1 - c_2 - c_3$$
* This completely attenuates frequencies shorter than the cut-off period while retaining near-zero phase lag.

### 3. Kaufman Efficiency Dynamic Blending
* Directional movement over 10 bars divided by the total sum of absolute bar changes yields directional efficiency $\in [0, 1]$.
* In clean trending environments ($ER \to 1.0$), up to 100% weight is given to the fast adaptive MA.
* In choppy, noisy environments ($ER \to 0.0$), 70% weight is allocated to the Ehlers SuperSmoother line.

### 4. RMA Slope Noise Deadband & State Machine ($SQ$)
* Slope is computed as $Trend - Trend[1]$.
* To prevent whipsaws in ranging markets, an adaptive threshold is generated using Wilder's Running Moving Average (RMA with length 20) multiplied by 0.60.
* The state machine maintains the previous regime until price slope decisively breaks out beyond the threshold.
* Entries only trigger on the exact bar when $SQ$ flips to $+1$ (BUY) or $-1$ (SELL).

---

## 📊 Prop Challenge Architecture & Mathematical Rules

| Parameter | Value | Institutional Purpose |
| :--- | :--- | :--- |
| **Base Challenge Capital** | **$2,000.00** | Starting account reference equity. |
| **Target Profit** | **+8.0% (+$160.00)** | Target to achieve. Once reached, trading automatically halts to secure the pass. |
| **Max Total Drawdown** | **-8.0% (-$160.00)** | Hard floor at $1,840.00. Bot terminates if equity reaches this floor. |
| **Daily Loss Limit** | **-5.0% (-$100.00)** | Hard daily loss ceiling. |
| **Daily Circuit Breaker** | **-3.5% (-$70.00)** | Soft circuit stop: Bot pauses taking new trades if daily loss hits -$70 to avoid slippage breaching the hard 5% limit. |
| **Max Daily Losses** | **2 Losses** | Halts trading for the rest of the day if 2 losses occur ($50 max loss, well below $70 breaker). |
| **Risk Per Trade** | **1.25% ($25.00)** | Dynamic lot sizing guarantees maximum dollar loss per trade is exactly $25. |
| **Risk-to-Reward (R:R)** | **1 : 2** | Risk $25 to make $50 per trade. |
| **Target Requirement** | **4 net wins** | With 1:2 R:R, just 4 net winning trades achieves the +$160 challenge pass. |

---

## 🖥️ Live Web Dashboard Features

* **Live KPIs:** Real-time challenge target progress (+8% meter), Daily loss safety gauge, Challenge equity ($2k baseline), and OANDA account NAV.
* **Indicator Regime Station:** Live display of Adaptive SuperSmoother trend line, slope, RMA threshold, and Kaufman efficiency for all configured pairs.
* **Interactive Equity Curve:** Chart.js plotting actual equity against the $2,160 Target line and $1,840 Stop line.
* **Active Positions:** Displays open trades, ticket IDs, unrealized PnL, pip gain/loss, with individual "Close" buttons.
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
cd stategy4
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
http://localhost:5003
```

---

## 📈 Historical Backtesting

To test the Adaptive SuperSmoother strategy against historical OANDA candlestick data:

```bash
cd stategy4
python backtest.py --symbol GBP_USD --count 500 --rr 2.0
python backtest.py --symbol EUR_USD --count 500 --rr 2.0
```

---

## 🌐 100% FREE 24/5 Deployment to Render

You can host this bot on Render completely free without a credit card.

### Step 1: Push Code to GitHub
1. Create a repository on GitHub (or use your existing repository).
2. Commit and push the `stategy4` directory.

### Step 2: Deploy as a Free Web Service on Render
1. Go to [https://render.com](https://render.com) and sign up (Free, No Credit Card required).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository (specify root directory `stategy4` if deploying from a monorepo).
4. Fill in the settings:
   * **Name:** `oanda-strategy4-bot`
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
6. Click **Create Web Service**.

---

## ⏰ Keep It Running 24/5 for Free (UptimeRobot)

Render free web services sleep after 15 minutes of inactivity. To keep your trading bot scanning continuously:

1. Go to [https://uptimerobot.com](https://uptimerobot.com) and create a free account.
2. Click **Add New Monitor**:
   * **Monitor Type:** `HTTP(s)`
   * **Friendly Name:** `Strategy 4 Keep-Alive`
   * **URL:** `https://your-render-app-name.onrender.com/health`
   * **Monitoring Interval:** `Every 5 minutes`
3. Click **Create Monitor**.

UptimeRobot will ping `/health` every 5 minutes. This prevents Render from sleeping, keeping your bot active 24/5 at **$0.00 cost**!

