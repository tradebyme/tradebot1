import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from config import Config

logger = logging.getLogger("RiskManager5")


class RiskManager:
    """
    Manages Prop Firm Challenge compliance, Capital Allocation, and Position Sizing.
    Enforces daily loss circuit breakers (3%) and strict overall drawdown floors (8%-10%).
    """

    def __init__(self, state_file_path: Optional[str] = None):
        if state_file_path is None:
            self.state_file = Path(__file__).resolve().parent / "state.json"
        else:
            self.state_file = Path(state_file_path)

        self.virtual_capital = Config.VIRTUAL_CAPITAL
        self.target_profit_usd = self.virtual_capital * (Config.TARGET_PROFIT_PCT / 100.0)
        self.target_profit_p2_usd = self.virtual_capital * (Config.TARGET_PROFIT_P2_PCT / 100.0)
        self.max_drawdown_usd = self.virtual_capital * (Config.MAX_DRAWDOWN_PCT / 100.0)
        self.daily_loss_limit_usd = self.virtual_capital * (Config.MAX_DAILY_LOSS_PCT / 100.0)
        self.circuit_breaker_usd = self.virtual_capital * (Config.DAILY_CIRCUIT_BREAKER_PCT / 100.0)
        self.risk_per_trade_usd = self.virtual_capital * (Config.RISK_PER_TRADE_PCT / 100.0)

        self.state = self._load_state()

    def _get_current_utc_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _default_state(self) -> Dict[str, Any]:
        today = self._get_current_utc_date()
        return {
            "challenge_start_date": Config.CHALLENGE_START_DATE or today,
            "current_day_utc": today,
            "virtual_capital": self.virtual_capital,
            "starting_balance_today": self.virtual_capital,
            "daily_realized_pl": 0.0,
            "daily_trades_count": 0,
            "daily_losses_count": 0,
            "total_realized_pl": 0.0,
            "total_trades_count": 0,
            "total_wins": 0,
            "total_losses": 0,
            "current_phase": "PHASE_1",
            "status": "ACTIVE",  # ACTIVE, PAUSED, TARGET_HIT, DAILY_LIMIT_HIT, MAX_DRAWDOWN_HIT
            "status_message": "Bot active and monitoring macro trend.",
            "equity_history": [
                {"time": datetime.now(timezone.utc).isoformat(), "equity": self.virtual_capital}
            ],
            "last_update": datetime.now(timezone.utc).isoformat()
        }

    def _load_state(self) -> Dict[str, Any]:
        """Load state from state.json or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    # Check if day has rolled over
                    today = self._get_current_utc_date()
                    if data.get("current_day_utc") != today:
                        logger.info(f"New UTC day detected ({today}). Resetting daily risk metrics.")
                        data["current_day_utc"] = today
                        data["starting_balance_today"] = data["virtual_capital"] + data.get("total_realized_pl", 0.0)
                        data["daily_realized_pl"] = 0.0
                        data["daily_trades_count"] = 0
                        data["daily_losses_count"] = 0
                        if data["status"] in ("DAILY_LIMIT_HIT", "PAUSED"):
                            data["status"] = "ACTIVE"
                            data["status_message"] = "Daily limits refreshed for new UTC day."
                    return data
            except Exception as e:
                logger.error(f"Error reading state file {self.state_file}: {e}. Creating new state.")

        state = self._default_state()
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[Dict[str, Any]] = None):
        """Persist state to disk."""
        target_state = state or self.state
        target_state["last_update"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(target_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    def check_new_day(self):
        """Check if UTC day has rolled over and reset daily counters."""
        today = self._get_current_utc_date()
        if self.state.get("current_day_utc") != today:
            logger.info(f"Rolling over to new UTC day: {today}")
            self.state["current_day_utc"] = today
            current_equity = self.virtual_capital + self.state.get("total_realized_pl", 0.0)
            self.state["starting_balance_today"] = current_equity
            self.state["daily_realized_pl"] = 0.0
            self.state["daily_trades_count"] = 0
            self.state["daily_losses_count"] = 0
            if self.state["status"] == "DAILY_LIMIT_HIT":
                self.state["status"] = "ACTIVE"
                self.state["status_message"] = "Daily limit cleared. Resuming trading for new day."
            self._save_state()

    def can_open_trade(self, current_open_trades_count: int) -> Tuple[bool, str]:
        """
        Verify if opening a new trade complies with Prop Firm Rules & Internal Circuit Breakers.
        """
        self.check_new_day()

        if self.state["status"] == "MAX_DRAWDOWN_HIT":
            return False, "Trading halted: Maximum Drawdown breached. Account protected."

        if self.state["status"] == "DAILY_LIMIT_HIT":
            return False, "Trading halted: Daily Loss Limit reached. Preserving account until tomorrow."

        if self.state["status"] == "TARGET_HIT":
            return False, "Trading paused: Challenge Profit Target achieved!"

        if self.state["status"] == "PAUSED":
            return False, "Trading manually paused."

        # Concurrent trades cap
        if current_open_trades_count >= Config.MAX_CONCURRENT_TRADES:
            return False, f"Maximum concurrent positions ({Config.MAX_CONCURRENT_TRADES}) reached."

        # Daily trades cap
        if self.state.get("daily_trades_count", 0) >= Config.MAX_TRADES_PER_DAY:
            return False, f"Maximum daily trades ({Config.MAX_TRADES_PER_DAY}) reached."

        # Daily consecutive loss cap
        if self.state.get("daily_losses_count", 0) >= Config.MAX_DAILY_LOSSES:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"Circuit Breaker Triggered: {Config.MAX_DAILY_LOSSES} losses today."
            self._save_state()
            return False, "Daily circuit breaker triggered: 2 consecutive losses."

        # Daily loss circuit breaker ($60 / 3% on $2,000)
        daily_loss = -self.state.get("daily_realized_pl", 0.0)
        if daily_loss >= self.circuit_breaker_usd:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"Circuit Breaker: Daily loss (${daily_loss:.2f}) reached 3.0% threshold."
            self._save_state()
            return False, f"Daily circuit breaker breached (${daily_loss:.2f} >= ${self.circuit_breaker_usd:.2f})."

        # Overall Max Drawdown ($160 / 8% on $2,000)
        total_loss = -self.state.get("total_realized_pl", 0.0)
        if total_loss >= self.max_drawdown_usd:
            self.state["status"] = "MAX_DRAWDOWN_HIT"
            self.state["status_message"] = f"Hard Floor Breached: Total loss (${total_loss:.2f}) >= Max Drawdown (${self.max_drawdown_usd:.2f})."
            self._save_state()
            return False, "Maximum Drawdown breached. Trading locked."

        return True, "Risk checks passed. Trade allowed."

    def calculate_position_size(self, instrument: str, entry_price: float, stop_loss_price: float) -> int:
        """
        Calculate exact order units so that the loss at stop_loss matches the Risk Per Trade budget.
        Budget: $15.00 on $2,000 (0.75%) or $10.00 on $2,000 (0.50%).
        """
        price_dist = abs(entry_price - stop_loss_price)
        if price_dist <= 0:
            logger.warning(f"Invalid stop distance {price_dist}. Using safe default 1000 units.")
            return 1000

        inst = instrument.upper()
        # For Gold (XAU_USD): 1 unit = 1 troy oz. Price move of $1.00 on 1 unit = $1.00 USD.
        if "XAU" in inst or "GOLD" in inst:
            units = int(self.risk_per_trade_usd / price_dist)
            return max(1, units)

        # For USD/JPY: 1 unit move of 1.00 yen = 1 / 150 USD approx.
        elif "JPY" in inst:
            jpy_rate = entry_price if entry_price > 50 else 150.0
            usd_loss_per_unit = price_dist / jpy_rate
            units = int(self.risk_per_trade_usd / usd_loss_per_unit) if usd_loss_per_unit > 0 else 1000
            return max(100, units)

        # For EUR/USD, GBP/USD: Quote currency is USD. 1 unit move of $0.0001 = $0.0001 USD.
        else:
            units = int(self.risk_per_trade_usd / price_dist)
            return max(100, units)

    def record_trade_result(self, trade_id: str, instrument: str, profit_loss: float):
        """Record the realized outcome of a closed trade and update challenge state."""
        self.check_new_day()

        self.state["total_realized_pl"] += profit_loss
        self.state["daily_realized_pl"] += profit_loss
        self.state["total_trades_count"] += 1
        self.state["daily_trades_count"] += 1

        if profit_loss >= 0:
            self.state["total_wins"] += 1
        else:
            self.state["total_losses"] += 1
            self.state["daily_losses_count"] += 1

        current_equity = self.virtual_capital + self.state["total_realized_pl"]

        # Append to equity curve history (limit to last 300 points)
        self.state["equity_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2),
            "trade_id": trade_id,
            "instrument": instrument,
            "pl": round(profit_loss, 2)
        })
        if len(self.state["equity_history"]) > 300:
            self.state["equity_history"] = self.state["equity_history"][-300:]

        # Check Challenge Status
        if self.state["total_realized_pl"] >= self.target_profit_usd:
            self.state["status"] = "TARGET_HIT"
            self.state["status_message"] = f"CONGRATULATIONS! Target (+${self.target_profit_usd:.2f}) Achieved! Challenge Passed."
        elif -self.state["total_realized_pl"] >= self.max_drawdown_usd:
            self.state["status"] = "MAX_DRAWDOWN_HIT"
            self.state["status_message"] = f"CRITICAL: Max Drawdown (-${self.max_drawdown_usd:.2f}) reached."
        elif -self.state["daily_realized_pl"] >= self.circuit_breaker_usd:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"Daily Circuit Breaker Triggered (-${self.circuit_breaker_usd:.2f})."

        self._save_state()
        logger.info(
            f"Recorded trade #{trade_id} ({instrument}) P/L: ${profit_loss:+.2f} | "
            f"Daily P/L: ${self.state['daily_realized_pl']:+.2f} | Total Equity: ${current_equity:.2f}"
        )

    def get_dashboard_metrics(self, open_unrealized_pl: float = 0.0) -> Dict[str, Any]:
        """Aggregate all metrics for real-time Web Dashboard display."""
        realized_pl = self.state.get("total_realized_pl", 0.0)
        total_pl = realized_pl + open_unrealized_pl
        current_equity = self.virtual_capital + total_pl
        daily_pl = self.state.get("daily_realized_pl", 0.0) + open_unrealized_pl

        total_trades = self.state.get("total_trades_count", 0)
        wins = self.state.get("total_wins", 0)
        win_rate = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0.0

        target = self.target_profit_usd
        progress_pct = max(0.0, min(100.0, (total_pl / target) * 100.0)) if target > 0 else 0.0

        return {
            "virtual_capital": self.virtual_capital,
            "current_equity": round(current_equity, 2),
            "realized_pl": round(realized_pl, 2),
            "floating_pl": round(open_unrealized_pl, 2),
            "daily_pl": round(daily_pl, 2),
            "target_profit_usd": target,
            "max_drawdown_usd": self.max_drawdown_usd,
            "daily_loss_limit_usd": self.daily_loss_limit_usd,
            "circuit_breaker_usd": self.circuit_breaker_usd,
            "risk_per_trade_usd": self.risk_per_trade_usd,
            "total_trades": total_trades,
            "total_wins": wins,
            "total_losses": self.state.get("total_losses", 0),
            "win_rate": win_rate,
            "progress_pct": round(progress_pct, 1),
            "status": self.state.get("status", "ACTIVE"),
            "status_message": self.state.get("status_message", "Bot active"),
            "current_phase": self.state.get("current_phase", "PHASE_1"),
            "equity_history": self.state.get("equity_history", [])
        }
