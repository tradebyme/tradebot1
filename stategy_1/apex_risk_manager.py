import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("ApexRiskManager")


class ApexRiskManager:
    """
    Manages Prop Firm Challenge compliance, Capital Allocation, and Position Sizing for APEX SNIPER.
    Enforces daily loss circuit breakers (3%) and strict overall drawdown floors (8%-10%).
    Persists state in state_apex.json.
    """

    def __init__(self, state_file_path: Optional[str] = None):
        if state_file_path is None:
            self.state_file = Path(__file__).resolve().parent / "state_apex.json"
        else:
            self.state_file = Path(state_file_path)

        self.virtual_capital = float(os.getenv("VIRTUAL_CAPITAL", "5000.0"))
        self.target_profit_usd = self.virtual_capital * (float(os.getenv("TARGET_PROFIT_PCT", "8.0")) / 100.0)
        self.target_profit_p2_usd = self.virtual_capital * (float(os.getenv("TARGET_PROFIT_P2_PCT", "5.0")) / 100.0)
        self.max_drawdown_usd = self.virtual_capital * (float(os.getenv("MAX_DRAWDOWN_PCT", "8.0")) / 100.0)
        self.daily_loss_limit_usd = self.virtual_capital * (float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0")) / 100.0)
        self.circuit_breaker_usd = self.virtual_capital * (float(os.getenv("DAILY_CIRCUIT_BREAKER_PCT", "3.0")) / 100.0)
        self.risk_per_trade_usd = self.virtual_capital * (float(os.getenv("APEX_RISK_PER_TRADE_PCT", "0.75")) / 100.0)

        self.state = self._load_state()

    def _get_current_utc_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _default_state(self) -> Dict[str, Any]:
        today = self._get_current_utc_date()
        return {
            "challenge_start_date": os.getenv("CHALLENGE_START_DATE", today),
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
            "status_message": "APEX SNIPER active and monitoring macro trend.",
            "equity_history": [
                {"time": datetime.now(timezone.utc).isoformat(), "equity": self.virtual_capital}
            ],
            "last_update": datetime.now(timezone.utc).isoformat()
        }

    def _load_state(self) -> Dict[str, Any]:
        """Load state from state_apex.json or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    today = self._get_current_utc_date()
                    if data.get("current_day_utc") != today:
                        logger.info(f"New UTC day detected ({today}). Resetting daily risk metrics.")
                        data["current_day_utc"] = today
                        data["daily_realized_pl"] = 0.0
                        data["daily_trades_count"] = 0
                        data["daily_losses_count"] = 0
                        data["starting_balance_today"] = data["virtual_capital"] + data["total_realized_pl"]
                        if data.get("status") == "DAILY_LIMIT_HIT":
                            data["status"] = "ACTIVE"
                            data["status_message"] = "Daily circuit breaker reset for new trading day."
                        self._save_state(data)
                    return data
            except Exception as e:
                logger.error(f"Error loading state file: {e}. Generating fresh state.")

        fresh = self._default_state()
        self._save_state(fresh)
        return fresh

    def _save_state(self, state: Dict[str, Any]):
        """Safely write state to state_apex.json."""
        try:
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def can_open_new_trade(self, current_open_count: int, max_concurrent: int, max_daily_trades: int) -> Tuple[bool, str]:
        """Verify strict prop firm compliance rules before submitting any order."""
        status = self.state.get("status", "ACTIVE")
        if status == "PAUSED":
            return False, "Bot is manually paused by operator."
        if status == "DAILY_LIMIT_HIT":
            return False, "Daily circuit breaker tripped (3.0%). Trading halted until tomorrow."
        if status == "MAX_DRAWDOWN_HIT":
            return False, "Maximum allowable drawdown breached. Account protected."
        if status == "TARGET_HIT":
            return False, f"Target profit reached! {self.state.get('current_phase')} Passed!"

        if current_open_count >= max_concurrent:
            return False, f"Maximum concurrent positions reached ({current_open_count}/{max_concurrent})."

        daily_count = self.state.get("daily_trades_count", 0)
        if daily_count >= max_daily_trades:
            return False, f"Daily trade limit reached ({daily_count}/{max_daily_trades})."

        daily_losses = self.state.get("daily_losses_count", 0)
        if daily_losses >= int(os.getenv("MAX_DAILY_LOSSES", "2")):
            return False, f"Max daily consecutive losses reached ({daily_losses})."

        return True, "Risk check passed."

    def calculate_position_size(self, entry_price: float, stop_loss: float, pip_value_usd_per_unit: float) -> int:
        """Calculate dynamic unit position size risking strictly RISK_PER_TRADE_USD."""
        dist = abs(entry_price - stop_loss)
        if dist <= 0 or pip_value_usd_per_unit <= 0:
            return 1000

        risk_usd = self.risk_per_trade_usd
        loss_per_unit = dist * pip_value_usd_per_unit
        units = int(risk_usd / loss_per_unit)
        return max(1, units)

    def record_closed_trade(self, trade_id: str, instrument: str, realized_pl: float):
        """Update metrics and circuit breakers when a trade closes."""
        self.state["total_realized_pl"] += realized_pl
        self.state["daily_realized_pl"] += realized_pl
        self.state["total_trades_count"] += 1
        self.state["daily_trades_count"] += 1

        if realized_pl > 0:
            self.state["total_wins"] += 1
        else:
            self.state["total_losses"] += 1
            self.state["daily_losses_count"] += 1

        current_equity = self.virtual_capital + self.state["total_realized_pl"]
        self.state["equity_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2),
            "trade_id": trade_id,
            "instrument": instrument,
            "pl": round(realized_pl, 2)
        })

        if -self.state["daily_realized_pl"] >= self.circuit_breaker_usd:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"Daily Circuit Breaker Tripped (-${abs(self.state['daily_realized_pl']):.2f}). Trading halted."
            logger.warning(self.state["status_message"])
        elif -self.state["total_realized_pl"] >= self.max_drawdown_usd:
            self.state["status"] = "MAX_DRAWDOWN_HIT"
            self.state["status_message"] = f"Max Drawdown Hit (-${abs(self.state['total_realized_pl']):.2f}). Trading halted."
            logger.error(self.state["status_message"])
        elif self.state["total_realized_pl"] >= self.target_profit_usd:
            self.state["status"] = "TARGET_HIT"
            self.state["status_message"] = f"Target Profit Achieved (+${self.state['total_realized_pl']:.2f})!"
            logger.info(self.state["status_message"])

        self._save_state(self.state)

    def set_pause(self, paused: bool):
        self.state["status"] = "PAUSED" if paused else "ACTIVE"
        self.state["status_message"] = "Bot paused by operator." if paused else "Bot resumed and monitoring macro trend."
        self._save_state(self.state)

    def reset_challenge(self):
        fresh = self._default_state()
        self.state = fresh
        self._save_state(self.state)

    def get_dashboard_metrics(self, floating_pl: float = 0.0) -> Dict[str, Any]:
        """Compile status metrics for the web dashboard."""
        realized = self.state.get("total_realized_pl", 0.0)
        daily_pl = self.state.get("daily_realized_pl", 0.0)
        current_equity = self.virtual_capital + realized + floating_pl

        progress = max(0.0, min(100.0, (realized / self.target_profit_usd) * 100.0)) if self.target_profit_usd > 0 else 0.0
        total_trades = self.state.get("total_trades_count", 0)
        wins = self.state.get("total_wins", 0)
        win_rate = round((wins / total_trades) * 100.0, 1) if total_trades > 0 else 0.0

        return {
            "virtual_capital": self.virtual_capital,
            "current_equity": round(current_equity, 2),
            "realized_pl": round(realized, 2),
            "floating_pl": round(floating_pl, 2),
            "daily_pl": round(daily_pl, 2),
            "target_profit_usd": self.target_profit_usd,
            "progress_pct": round(progress, 1),
            "max_drawdown_usd": self.max_drawdown_usd,
            "circuit_breaker_usd": self.circuit_breaker_usd,
            "status": self.state.get("status", "ACTIVE"),
            "status_message": self.state.get("status_message", ""),
            "total_trades": total_trades,
            "total_wins": wins,
            "total_losses": self.state.get("total_losses", 0),
            "win_rate": win_rate,
            "current_phase": self.state.get("current_phase", "PHASE_1"),
            "equity_history": self.state.get("equity_history", [])
        }
