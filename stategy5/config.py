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
    """
    Strategy 5: Institutional Trend-ATR Portfolio (ITAP) Engine
    Designed for Prop Firm Challenge Compliance & Cloud Deployment on Render.
    """

    # 1. OANDA API Credentials
    # Check bot-specific key first (OANDA_API_KEY_5 / OANDA_API_KEY_APEX), then global key
    API_KEY = (
        os.getenv("OANDA_API_KEY_5")
        or os.getenv("OANDA_API_KEY_APEX")
        or os.getenv("OANDA_API_KEY")
        or os.getenv("oanda_api")
        or os.getenv("OANDA_API", "7440c485ceedec42c38b0b49f61b7992-88ff981ae8caf36fd9e26fdfde387992")
    ).strip()
    ACCOUNT_ID = (
        os.getenv("OANDA_ACCOUNT_ID_5")
        or os.getenv("OANDA_ACCOUNT_ID_APEX")
        or os.getenv("OANDA_ACCOUNT_ID")
        or os.getenv("oanda_account_id", "101-001-41055093-001")
    ).strip()
    ENVIRONMENT = os.getenv("OANDA_ENV", "practice").lower().strip()  # 'practice' or 'live'

    # OANDA API Endpoints
    if ENVIRONMENT == "live":
        BASE_URL = "https://api-fxtrade.oanda.com/v3"
        STREAM_URL = "https://stream-fxtrade.oanda.com/v3"
    else:
        BASE_URL = "https://api-fxpractice.oanda.com/v3"
        STREAM_URL = "https://stream-fxpractice.oanda.com/v3"

    # 2. Prop Challenge & Capital Rules ($2,000 / $5,000 / $10,000)
    CHALLENGE_START_DATE = os.getenv("CHALLENGE_START_DATE", "2026-11-15").strip()
    VIRTUAL_CAPITAL = float(os.getenv("VIRTUAL_CAPITAL", "2000.0"))
    TARGET_PROFIT_PCT = float(os.getenv("TARGET_PROFIT_PCT", "8.0"))       # Phase 1 Target: 8%
    TARGET_PROFIT_P2_PCT = float(os.getenv("TARGET_PROFIT_P2_PCT", "5.0")) # Phase 2 Target: 5%
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))         # Max Overall Drawdown Floor: 8% - 10%
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))     # Prop Firm Daily Loss Limit: 5.0%

    # Proactive Institutional Circuit Breakers
    DAILY_CIRCUIT_BREAKER_PCT = float(os.getenv("DAILY_CIRCUIT_BREAKER_PCT", "3.0"))  # Halts if daily loss >= 3.0%
    MAX_DAILY_LOSSES = int(os.getenv("MAX_DAILY_LOSSES", "2"))  # Halts trading after 2 losses in a day

    # 3. Strategy Core Configuration (Macro Trend + Momentum Crossover + ATR)
    FAST_EMA = int(os.getenv("FAST_EMA", "20"))
    SLOW_EMA = int(os.getenv("SLOW_EMA", "50"))
    TREND_EMA = int(os.getenv("TREND_EMA", "200"))
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

    # Multipliers
    ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "2.0"))        # Initial SL = 2.0x ATR
    ATR_TRAIL_MULTIPLIER = float(os.getenv("ATR_TRAIL_MULTIPLIER", "3.0"))  # Trailing Stop = 3.0x ATR
    MAX_DIST_ATR_MULTIPLIER = float(os.getenv("MAX_DIST_ATR_MULTIPLIER", "2.0"))  # Anti-exhaustion filter: < 2.0x ATR

    # Trailing Stop Ratchet Settings
    TRAIL_ACTIVATION_ATR = float(os.getenv("TRAIL_ACTIVATION_ATR", "1.0"))  # Start ratcheting after +1.0 ATR in profit

    # 4. Portfolio Instruments & Multi-Timeframe Mapping
    # Portfolio basket: XAU_USD (Gold H4), EUR_USD (H4), GBP_USD (H1)
    INSTRUMENTS = [i.strip() for i in os.getenv("INSTRUMENTS", "XAU_USD,EUR_USD,GBP_USD").split(",") if i.strip()]
    TIMEFRAME_MAPPING = {
        "XAU_USD": os.getenv("TF_XAUUSD", "H4"),
        "EUR_USD": os.getenv("TF_EURUSD", "H4"),
        "GBP_USD": os.getenv("TF_GBPUSD", "H1"),
        "USD_JPY": os.getenv("TF_USDJPY", "H4")
    }
    DEFAULT_TIMEFRAME = os.getenv("TIMEFRAME", "H4")

    # 5. Position Sizing & Risk per Trade
    # 0.75% for Evaluation Sprint; 0.50% for Funded Account Preservation
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.75"))
    MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "2"))
    MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
    MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "4.0"))  # 4 pips Gold / 2 pips FX

    # 6. Session Filter (Optional - Default False for H4 macro swing)
    SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "false").lower() in ("true", "1", "yes")
    SESSION_START_HOUR_UTC = int(os.getenv("SESSION_START_HOUR_UTC", "6"))
    SESSION_END_HOUR_UTC = int(os.getenv("SESSION_END_HOUR_UTC", "20"))

    # 7. Web Dashboard Server & Security
    PORT = int(os.getenv("PORT", "5005"))
    HOST = os.getenv("HOST", "0.0.0.0")
    SECRET_KEY = os.getenv("SECRET_KEY", "strategy5-itap-cloud-secret-key-2026")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin").strip()
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "oandabot2026").strip()
