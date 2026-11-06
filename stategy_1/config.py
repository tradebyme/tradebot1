import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current dir or parent dir
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

if (current_dir / ".env").exists():
    load_dotenv(current_dir / ".env")
elif (parent_dir / ".env").exists():
    load_dotenv(parent_dir / ".env")
else:
    load_dotenv()

class Config:
    # OANDA Credentials
    # Accept OANDA_API_KEY, oanda_api, or OANDA_API
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

    # Prop Challenge & Capital Rules (User Target: 8% gain in 3 days, 5% daily loss limit, 8% max loss)
    CHALLENGE_START_DATE = os.getenv("CHALLENGE_START_DATE", "2026-11-05").strip()
    VIRTUAL_CAPITAL = float(os.getenv("VIRTUAL_CAPITAL", "2000.0"))
    TARGET_PROFIT_PCT = float(os.getenv("TARGET_PROFIT_PCT", "8.0"))       # 8% target ($160 on $2,000)
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))         # 8% max total loss ($160 on $2,000)
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))     # 5% max daily loss ($100 on $2,000)
    
    # Proactive Circuit Breakers
    DAILY_CIRCUIT_BREAKER_PCT = float(os.getenv("DAILY_CIRCUIT_BREAKER_PCT", "3.5")) # $70 on $2,000
    MAX_DAILY_LOSSES = int(os.getenv("MAX_DAILY_LOSSES", "2")) # Halts trading if 2 losses occur in 1 day (0% drawdown failure in 2-yr backtest)

    # Risk Management Per Trade
    # 1.25% risk per trade ($25 on $2,000)
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.25"))
    # 1:2 Risk to Reward (Risk $25 to make $50)
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))

    # Strategy Configuration
    # Primary focus pairs: GBP_USD (proven +$2,400 profit in 2-year test), EUR_USD
    INSTRUMENTS = [i.strip() for i in os.getenv("INSTRUMENTS", "GBP_USD,EUR_USD").split(",") if i.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "M15") # 15-Minute candles
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "1"))
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
    MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "6.0"))

    # Session Filter (London: 07:00 UTC - NY Overlap: 17:00 UTC)
    SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")
    SESSION_START_HOUR_UTC = int(os.getenv("SESSION_START_HOUR_UTC", "7"))
    SESSION_END_HOUR_UTC = int(os.getenv("SESSION_END_HOUR_UTC", "17"))

    # Server & Security
    PORT = int(os.getenv("PORT", "5000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    SECRET_KEY = os.getenv("SECRET_KEY", "oanda-super-secret-key-change-me")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin").strip()
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "oandabot2026").strip()

    # Bot Loop Polling Interval (in seconds)
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))

    @classmethod
    def validate(cls):
        """Validate critical configuration parameters."""
        if not cls.API_KEY:
            raise ValueError(
                "OANDA API Key is missing! Please set OANDA_API_KEY or oanda_api in your .env file."
            )
        return True

