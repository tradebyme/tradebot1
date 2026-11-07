import os
import sys
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Load .env from current dir or parent dir
if (current_dir / ".env").exists():
    load_dotenv(current_dir / ".env")
elif (parent_dir / ".env").exists():
    load_dotenv(parent_dir / ".env")
else:
    load_dotenv()

class Config:
    """Configuration for Strategy 3: ICT / SMC Asian Liquidity Sweep & Judas Swing."""

    # OANDA Credentials
    API_KEY = os.getenv("OANDA_API_KEY") or os.getenv("oanda_api") or os.getenv("OANDA_API", "").strip()
    ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("oanda_account_id", "").strip()
    ENVIRONMENT = os.getenv("OANDA_ENV", "practice").lower().strip() # 'practice' or 'live'

    # OANDA Endpoints
    if ENVIRONMENT == "live":
        BASE_URL = "https://api-fxtrade.oanda.com/v3"
        STREAM_URL = "https://stream-fxtrade.oanda.com/v3"
    else:
        BASE_URL = "https://api-fxpractice.oanda.com/v3"
        STREAM_URL = "https://stream-fxpractice.oanda.com/v3"

    # Prop Challenge & Risk Rules ($2,000 Capital Target)
    CHALLENGE_START_DATE = os.getenv("CHALLENGE_START_DATE", "2026-11-07").strip()
    VIRTUAL_CAPITAL = float(os.getenv("VIRTUAL_CAPITAL", "2000.0"))
    TARGET_PROFIT_PCT = float(os.getenv("TARGET_PROFIT_PCT", "8.0"))       # +8% ($160)
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))         # -8% ($160)
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))     # -5% ($100)
    
    # Proactive Circuit Breakers
    DAILY_CIRCUIT_BREAKER_PCT = float(os.getenv("DAILY_CIRCUIT_BREAKER_PCT", "3.5")) # -$70
    MAX_DAILY_LOSSES = int(os.getenv("MAX_DAILY_LOSSES", "2")) # Halts after 2 losses in 1 day

    # Position Sizing & Reward-to-Risk
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.25")) # $25 on $2,000
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.5"))   # 1:1.5 R:R for >50% win rate (Target $37.50)

    # Strategy Parameters: ICT / SMC Liquidity Sweep
    ASIAN_SESSION_START_UTC = int(os.getenv("ASIAN_SESSION_START_UTC", "0"))   # 00:00 UTC
    ASIAN_SESSION_END_UTC = int(os.getenv("ASIAN_SESSION_END_UTC", "7"))       # 06:45 UTC (ends right before 07:00)
    LONDON_KILLZONE_START_UTC = int(os.getenv("LONDON_KILLZONE_START_UTC", "7")) # 07:00 UTC
    LONDON_KILLZONE_END_UTC = int(os.getenv("LONDON_KILLZONE_END_UTC", "11"))    # 11:00 UTC
    
    MIN_RANGE_PIPS = float(os.getenv("MIN_RANGE_PIPS", "8.0"))   # Skip dead sessions
    MAX_RANGE_PIPS = float(os.getenv("MAX_RANGE_PIPS", "50.0"))  # Skip high-impact news spikes
    SWEEP_BUFFER_PIPS = float(os.getenv("SWEEP_BUFFER_PIPS", "1.5")) # Pip buffer beyond sweep wick for SL

    # Instruments & Timeframe
    INSTRUMENTS = [i.strip() for i in os.getenv("INSTRUMENTS", "EUR_USD,GBP_USD,XAU_USD").split(",") if i.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "M15")
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "1"))
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
    MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "5.0"))

    # Server & Dashboard (Port 5002)
    PORT = int(os.getenv("PORT", "5002"))
    HOST = os.getenv("HOST", "0.0.0.0")
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
    SECRET_KEY = os.getenv("SECRET_KEY", "strategy3-smc-sweep-key-2026")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "oandabot2026")
