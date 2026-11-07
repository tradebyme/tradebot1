import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple
from config import Config

logger = logging.getLogger("SMC_RiskManager")

class RiskManager:
    """
    Institutional Prop Firm Risk & Compliance Manager for Strategy 3:
    - Starting Capital: $2,000.00
    - Fixed Risk per trade: $25.00 (1.25%)
    - Max 2 losses per day circuit breaker ($50 max loss per day, safely below 5% limit)
    - Target: +8% ($160.00)
    - Max Total Drawdown: 8% ($160.00)
    - UTC Midnight rollover & state persistence
    """

    def __init__(self, state_file: str = "state.json"):
        self.state_file = Path(__file__).resolve().parent / state_file
        self.virtual_capital = Config.VIRTUAL_CAPITAL
        self.target_profit_pct = Config.TARGET_PROFIT_PCT
        self.max_drawdown_pct = Config.MAX_DRAWDOWN_PCT
        self.max_daily_loss_pct = Config.MAX_DAILY_LOSS_PCT
        self.risk_per_trade_pct = Config.RISK_PER_TRADE_PCT
        self.max_daily_losses = Config.MAX_DAILY_LOSSES

        self.state = {
            "virtual_capital": self.virtual_capital,
            "starting_balance": self.virtual_capital,
            "peak_balance": self.virtual_capital,
            "current_balance": self.virtual_capital,
            "net_pnl": 0.0,
            "today_pnl": 0.0,
            "daily_loss_count": 0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "current_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "daily_circuit_breaker_active": False,
            "challenge_status": "ACTIVE", # 'ACTIVE', 'PASSED', 'FAILED'
            "status_reason": "Challenge in progress.",
            "trade_history": []
        }
        self.load_state()

    def load_state(self):
        """Load state from local json file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    saved = json.load(f)
                    self.state.update(saved)
                logger.info(f"Loaded existing risk state. Balance: ${self.state['current_balance']:,.2f}")
            except Exception as e:
                logger.error(f"Failed to load state.json: {e}")
        self.check_daily_rollover()

    def save_state(self):
        """Persist state to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state.json: {e}")

    def check_daily_rollover(self):
        """Reset daily PnL and loss counter at 00:00 UTC."""
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.get("current_day") != utc_now:
            logger.info(f"Daily rollover: Resetting day metrics from {self.state.get('current_day')} to {utc_now}.")
            self.state["current_day"] = utc_now
            self.state["today_pnl"] = 0.0
            self.state["daily_loss_count"] = 0
            self.state["daily_circuit_breaker_active"] = False
            self.save_state()

    def can_open_trade(self) -> Tuple[bool, str]:
        """Verify all prop firm risk parameters before placing order."""
        self.check_daily_rollover()

        # 1. Challenge status
        if self.state["challenge_status"] == "FAILED":
            return False, f"Trading halted: Challenge failed ({self.state['status_reason']})."
        if self.state["challenge_status"] == "PASSED":
            return False, "Trading halted: Challenge goal reached! Target achieved."

        # 2. Daily circuit breaker: 2 losses
        if self.state["daily_loss_count"] >= self.max_daily_losses:
            self.state["daily_circuit_breaker_active"] = True
            self.save_state()
            return False, f"Daily Circuit Breaker Triggered: Max {self.max_daily_losses} daily losses reached. Trading halted until 00:00 UTC."

        # 3. Daily loss ceiling ($100 / 5%)
        max_daily_loss_usd = self.virtual_capital * (self.max_daily_loss_pct / 100.0)
        if self.state["today_pnl"] <= -max_daily_loss_usd:
            self.state["daily_circuit_breaker_active"] = True
            self.save_state()
            return False, f"Daily 5% loss ceiling hit (${self.state['today_pnl']:.2f} <= -${max_daily_loss_usd:.2f})."

        # 4. Total drawdown check ($160 / 8%)
        max_total_dd_usd = self.virtual_capital * (self.max_drawdown_pct / 100.0)
        current_dd = self.state["peak_balance"] - self.state["current_balance"]
        if current_dd >= max_total_dd_usd:
            self.state["challenge_status"] = "FAILED"
            self.state["status_reason"] = f"Max Total Drawdown Breached (${current_dd:.2f} >= ${max_total_dd_usd:.2f})."
            self.save_state()
            return False, self.state["status_reason"]

        return True, "Risk checks passed."

    def calculate_position_size(self, symbol: str, risk_pips: float, current_price: float) -> int:
        """
        Calculate exact position units for fixed $25 risk:
        Units = Dollars at Risk / (Stop Pips * Pip Value per unit)
        """
        risk_usd = self.virtual_capital * (self.risk_per_trade_pct / 100.0) # $25.00
        sym_upper = symbol.upper().replace("_", "")

        if risk_pips <= 0:
            return 1000

        if "XAU" in sym_upper or "GOLD" in sym_upper:
            # 1 standard lot = 100 oz. 1 pip = $0.10. 1 oz moves $0.10 = $0.10.
            # Pip value per unit (1 oz) = $0.10.
            units = int(risk_usd / (risk_pips * 0.10))
            return max(1, min(units, 20)) # Clamp safely
        elif "JPY" in sym_upper:
            # JPY pip value approx $0.065 to $0.09 per 1,000 units
            pip_val_per_unit = 0.01 / current_price
            units = int(risk_usd / (risk_pips * pip_val_per_unit))
            return max(1000, min(units, 50000))
        else: # Standard EUR/USD, GBP/USD
            # 1 pip = 0.0001 = $0.0001 per unit
            pip_val_per_unit = 0.0001
            units = int(risk_usd / (risk_pips * pip_val_per_unit))
            return max(1000, min(units, 30000))

    def record_closed_trade(self, trade_data: Dict[str, Any]):
        """Record trade result, update PnL, and check challenge status."""
        self.check_daily_rollover()
        pnl = float(trade_data.get("realized_pnl", 0.0))

        self.state["current_balance"] += pnl
        self.state["net_pnl"] += pnl
        self.state["today_pnl"] += pnl
        self.state["total_trades"] += 1

        if pnl > 0:
            self.state["wins"] += 1
        elif pnl < 0:
            self.state["losses"] += 1
            self.state["daily_loss_count"] += 1

        if self.state["current_balance"] > self.state["peak_balance"]:
            self.state["peak_balance"] = self.state["current_balance"]

        # Check circuit breaker
        if self.state["daily_loss_count"] >= self.max_daily_losses:
            self.state["daily_circuit_breaker_active"] = True

        # Check Target (+8% / +$160)
        target_usd = self.virtual_capital * (self.target_profit_pct / 100.0)
        if self.state["net_pnl"] >= target_usd:
            self.state["challenge_status"] = "PASSED"
            self.state["status_reason"] = f"Challenge Target (+${target_usd:.2f}) Achieved!"

        # Check Max DD (-8% / -$160)
        max_total_dd_usd = self.virtual_capital * (self.max_drawdown_pct / 100.0)
        current_dd = self.state["peak_balance"] - self.state["current_balance"]
        if current_dd >= max_total_dd_usd:
            self.state["challenge_status"] = "FAILED"
            self.state["status_reason"] = f"Max Total Drawdown Breached (-${current_dd:.2f})."

        self.state["trade_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pnl": pnl,
            "instrument": trade_data.get("instrument", ""),
            "direction": trade_data.get("direction", ""),
            "balance_after": self.state["current_balance"]
        })
        self.save_state()
