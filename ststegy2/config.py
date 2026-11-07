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
    """Configuration for Strategy 2: Adaptive SuperSmoother + Trend Target Ribbon."""

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
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))   # 1:2 R:R (Target $50)

    # Strategy Parameters: Trend Target Ribbon (ALMA + StDev)
    ALMA_LENGTH = int(os.getenv("ALMA_LENGTH", "34"))
    ALMA_OFFSET = float(os.getenv("ALMA_OFFSET", "0.85"))
    ALMA_SIGMA = float(os.getenv("ALMA_SIGMA", "6.0"))
    DEV_LENGTH = int(os.getenv("DEV_LENGTH", "34"))
    DEV_MULT = float(os.getenv("DEV_MULT", "0.65"))
    SLOPE_LENGTH = int(os.getenv("SLOPE_LENGTH", "3"))
    SLOPE_MIN = float(os.getenv("SLOPE_MIN", "0.08"))
    ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))

    # Strategy Parameters: Adaptive SuperSmoother (Ehlers + Kaufman)
    MA_MIN = int(os.getenv("MA_MIN", "14"))
    MA_MAX = int(os.getenv("MA_MAX", "50"))
    VOL_LENGTH = int(os.getenv("VOL_LENGTH", "9"))
    SMOOTH_LENGTH = int(os.getenv("SMOOTH_LENGTH", "3"))
    ER_LENGTH = int(os.getenv("ER_LENGTH", "10"))
    FAST_BIAS = float(os.getenv("FAST_BIAS", "0.30"))
    NOISE_LENGTH = int(os.getenv("NOISE_LENGTH", "20"))
    NOISE_MULT = float(os.getenv("NOISE_MULT", "0.60"))

    # Stop Loss & Risk Bounds (Swing Structure + ATR Clamp)
    STOP_LOOKBACK = int(os.getenv("STOP_LOOKBACK", "12"))
    MIN_STOP_ATR = float(os.getenv("MIN_STOP_ATR", "0.75"))
    MAX_STOP_ATR = float(os.getenv("MAX_STOP_ATR", "2.5"))

    # Instruments & Timeframe
    INSTRUMENTS = [i.strip() for i in os.getenv("INSTRUMENTS", "XAU_USD,EUR_USD,USD_JPY").split(",") if i.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "H1")
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "1"))
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
    MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "6.0"))

    # Session Hours (UTC: London 07:00 to NY 17:00)
    SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")
    SESSION_START_HOUR_UTC = int(os.getenv("SESSION_START_HOUR_UTC", "7"))
    SESSION_END_HOUR_UTC = int(os.getenv("SESSION_END_HOUR_UTC", "17"))

    # Server & Dashboard
    PORT = int(os.getenv("PORT", "5001")) # Port 5001 to run alongside strategy 1
    HOST = os.getenv("HOST", "0.0.0.0")
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
    SECRET_KEY = os.getenv("SECRET_KEY", "strategy2-oanda-key-2026")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "oandabot2026")

