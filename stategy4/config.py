import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current directory, parent directory, or system environment
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

if (current_dir / ".env").exists():
    load_dotenv(current_dir / ".env")
elif (parent_dir / ".env").exists():
    load_dotenv(parent_dir / ".env")
else:
    load_dotenv()


class Config:
    """Strategy 4 Configuration - Adaptive SuperSmoother Strategy (Pine Script Mirror)."""

    # OANDA Credentials (supports OANDA_API_KEY, oanda_api, or OANDA_API)
    API_KEY = os.getenv("OANDA_API_KEY") or os.getenv("oanda_api") or os.getenv("OANDA_API", "").strip()
    ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("oanda_account_id", "").strip()
    ENVIRONMENT = os.getenv("OANDA_ENV", "practice").lower().strip()  # 'practice' or 'live'

    # OANDA API Endpoints
    if ENVIRONMENT == "live":
        BASE_URL = "https://api-fxtrade.oanda.com/v3"
        STREAM_URL = "https://stream-fxtrade.oanda.com/v3"
    else:
        BASE_URL = "https://api-fxpractice.oanda.com/v3"
        STREAM_URL = "https://stream-fxpractice.oanda.com/v3"

    # Prop Challenge & Capital Rules ($2,000 baseline, +8% target, 8% max loss)
    CHALLENGE_START_DATE = os.getenv("CHALLENGE_START_DATE", "2026-11-09").strip()
    VIRTUAL_CAPITAL = float(os.getenv("VIRTUAL_CAPITAL", "2000.0"))
    TARGET_PROFIT_PCT = float(os.getenv("TARGET_PROFIT_PCT", "8.0"))       # 8% target ($160 on $2,000)
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))         # 8% max total loss ($160 on $2,000)
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))     # 5% max daily loss ($100 on $2,000)

    # Proactive Circuit Breakers
    DAILY_CIRCUIT_BREAKER_PCT = float(os.getenv("DAILY_CIRCUIT_BREAKER_PCT", "3.5"))  # $70 on $2,000
    MAX_DAILY_LOSSES = int(os.getenv("MAX_DAILY_LOSSES", "2"))  # Halts trading after 2 losses in a day

    # Strategy & Risk Settings (Exact match with Strategy Pine Script wrapper)
    TRADE_DIRECTION = os.getenv("TRADE_DIRECTION", "Long & Short").strip()  # "Long & Short", "Long Only", "Short Only"
    USE_SL = os.getenv("USE_SL", "true").lower() in ("true", "1", "yes")    # Use ATR Stop Loss (default true)
    USE_TP = os.getenv("USE_TP", "false").lower() in ("true", "1", "yes")   # Use ATR Take Profit (default false in Pine Script)
    ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))                         # atrLen = 14
    ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "2.0"))        # atrSL = 2.0x ATR
    ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", "3.0"))        # atrTP = 3.0x ATR
    CLOSE_ON_OPPOSITE_SIGNAL = os.getenv("CLOSE_ON_OPPOSITE_SIGNAL", "true").lower() in ("true", "1", "yes")  # Position reversal on trend flip

    # Position Sizing Mode ("Risk % of Equity" or "Fixed Units")
    SIZE_MODE = os.getenv("SIZE_MODE", "Risk % of Equity").strip()
    FIXED_QTY = float(os.getenv("FIXED_QTY", "10000.0"))                    # 10,000 units = 0.10 lot
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.25"))    # 1.25% risk per trade ($25 on $2,000)
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))

    # Strategy Configuration (Instruments & Execution)
    INSTRUMENTS = [i.strip() for i in os.getenv("INSTRUMENTS", "GBP_USD,EUR_USD").split(",") if i.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "M15")  # 15-Minute candles
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "1"))
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
    MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.0"))

    # Adaptive SuperSmoother Parameters (From Pine Script)
    MA_MIN = int(os.getenv("MA_MIN", "14"))                # Minimum MA Period for high volatility
    MA_MAX = int(os.getenv("MA_MAX", "50"))                # Maximum MA Period for low volatility
    VOL_LENGTH = int(os.getenv("VOL_LENGTH", "9"))         # Volatility Period (ta.stdev & ta.sma)
    SMOOTH_LENGTH = int(os.getenv("SMOOTH_LENGTH", "3"))   # SuperSmoother Length (Ehlers 2-pole)
    ER_LENGTH = int(os.getenv("ER_LENGTH", "10"))          # Kaufman Efficiency Ratio Length
    FAST_BIAS = float(os.getenv("FAST_BIAS", "0.30"))      # Minimum Fast Weight (0.0 to 1.0)
    NOISE_LENGTH = int(os.getenv("NOISE_LENGTH", "20"))    # Noise Length for slope RMA
    NOISE_MULT = float(os.getenv("NOISE_MULT", "0.60"))    # Noise Multiplier for slope threshold

    # Session Filter (London: 07:00 UTC - NY Overlap: 17:00 UTC)
    SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")
    SESSION_START_HOUR_UTC = int(os.getenv("SESSION_START_HOUR_UTC", "7"))
    SESSION_END_HOUR_UTC = int(os.getenv("SESSION_END_HOUR_UTC", "17"))

    # Dynamic Peak Trailing & Profit Lock (MFE Protection)
    PEAK_TRAIL_ENABLED = os.getenv("PEAK_TRAIL_ENABLED", "true").lower() in ("true", "1", "yes")
    PEAK_TRAIL_ACTIVATION_R = float(os.getenv("PEAK_TRAIL_ACTIVATION_R", "1.2"))   # Activate once profit >= +1.2R
    PEAK_TRAIL_ATR_MULT = float(os.getenv("PEAK_TRAIL_ATR_MULT", "0.6"))           # Trail behind peak by 0.6x ATR
    BREAKEVEN_ACTIVATION_R = float(os.getenv("BREAKEVEN_ACTIVATION_R", "1.0"))     # Move SL to entry + buffer at +1.0R
    BREAKEVEN_BUFFER_R = float(os.getenv("BREAKEVEN_BUFFER_R", "0.1"))             # +0.1R buffer to cover spread/commissions

    # Smart Trend Continuation Re-Entry
    REENTRY_ENABLED = os.getenv("REENTRY_ENABLED", "true").lower() in ("true", "1", "yes")
    REENTRY_MAX_BARS = int(os.getenv("REENTRY_MAX_BARS", "6"))                     # Max bars since last exit to allow re-entry
    REENTRY_MAX_ATTEMPTS = int(os.getenv("REENTRY_MAX_ATTEMPTS", "1"))             # Max consecutive re-entries

    # Web Dashboard Server & Security
    PORT = int(os.getenv("PORT", "5003"))
    HOST = os.getenv("HOST", "0.0.0.0")
    SECRET_KEY = os.getenv("SECRET_KEY", "oanda-adaptive-supersmoother-key-2026")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin").strip()
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "oandabot2026").strip()

    # Polling Loop Interval (in seconds)
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))

    @classmethod
    def validate(cls):
        """Validate critical configuration parameters."""
        if not cls.API_KEY:
            raise ValueError(
                "OANDA API Key is missing! Please set OANDA_API_KEY or oanda_api in your .env file."
            )
        return True
