import sys
import os
from pathlib import Path

# Add stategy_1 directory to python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

print("=== 1. TESTING STRATEGY 1 TRADING BOT ===")
from config import Config
print(f"Config Virtual Capital: ${Config.VIRTUAL_CAPITAL:,.2f}")
print(f"Config Challenge Start Date: {Config.CHALLENGE_START_DATE}")
print(f"Config Risk Per Trade: {Config.RISK_PER_TRADE_PCT}% (${Config.VIRTUAL_CAPITAL * Config.RISK_PER_TRADE_PCT / 100:.2f})")
print(f"Config Risk-Reward Ratio: 1:{Config.RISK_REWARD_RATIO}")

# Clean state.json to test cold boot from OANDA broker history
state_path = current_dir / "state.json"
if state_path.exists():
    state_path.unlink()

from trading_bot import TradingBot
bot = TradingBot()
summary = bot.risk_manager.get_summary()
print(f"Bot Initialized: Equity=${summary['current_equity']:,.2f}, Status={summary['status']}")
print(f"Attempt #{summary['current_attempt_number']} | Total Attempts: {summary['total_attempts']}")
print(f"Archived Past Attempts: {len(summary['archived_attempts'])}")
if summary["archived_attempts"]:
    a = summary["archived_attempts"][0]
    print(f"  Attempt #{a['attempt_number']}: {a['start_date']} to {a['end_date']} | Trades: {a['total_trades']} | PnL: ${a['realized_pl']} | Status: {a['status']}")

print("\n=== 2. TESTING SCAN PASS WITH H1 TREND FILTER ===")
scan_res = bot.scan_once()
print(f"Scan Completed: Processed {len(scan_res.get('analysis', []))} pairs.")
for item in scan_res.get("analysis", []):
    inst = item.get("instrument")
    sig = item.get("signal")
    h1_trend = item.get("h1_trend", "N/A")
    print(f"  {inst}: Signal={sig}, H1 Trend={h1_trend}, Price={item.get('price')}")

print("\n=== 3. TESTING FLASK APP AND DASHBOARDS ===")
from app import app
client = app.test_client()
with client.session_transaction() as sess:
    sess["authenticated"] = True

# Strategy 1 Status
r1 = client.get("/api/status")
print(f"/api/status -> HTTP {r1.status_code}")
assert r1.status_code == 200, f"Expected 200, got {r1.status_code}"
data1 = r1.get_json()
print(f"  Strategy 1 Capital: ${data1['risk']['virtual_capital']:,.2f}, Target: ${data1['risk']['target_profit_usd']:,.2f}, Floor: ${data1['risk']['max_drawdown_floor']:,.2f}")
print(f"  Current Attempt: #{data1['risk']['current_attempt_number']}, Status: {data1['risk']['status']}")

# APEX Status
r2 = client.get("/api/apex/status")
print(f"/api/apex/status -> HTTP {r2.status_code}")
assert r2.status_code == 200, f"Expected 200, got {r2.status_code}"
data2 = r2.get_json()
apex_metrics = data2.get("metrics", {})
print(f"  APEX Capital: ${apex_metrics.get('virtual_capital', 0.0):,.2f}, Status: {apex_metrics.get('status')}, Bot Running: {data2.get('is_running')}")

# Dashboard HTML rendering
r3 = client.get("/")
print(f"/ (Dashboard) -> HTTP {r3.status_code}")
assert r3.status_code == 200, f"Expected 200, got {r3.status_code}"
html_text = r3.get_data(as_text=True)
assert "$5,000" in html_text, "Missing $5,000 in dashboard HTML!"
assert "Challenge Attempts" in html_text, "Missing Challenge Attempts tab in dashboard HTML!"
print("  Dashboard HTML verified: contains $5,000 baseline and Challenge Attempts tab.")

print("\n>>> ALL LOCAL TESTS PASSED 100%! SYSTEM IS READY! <<<")
