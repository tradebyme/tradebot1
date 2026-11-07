import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from config import Config

logger = logging.getLogger("RiskManager2")

class RiskManager:
    """Manages Prop Firm Challenge compliance, Capital Allocation, and Position Sizing for Strategy 2."""

    def __init__(self, state_file_path: Optional[str] = None):
        if state_file_path is None:
            self.state_file = Path(__file__).resolve().parent / "state.json"
        else:
            self.state_file = Path(state_file_path)

        self.virtual_capital = Config.VIRTUAL_CAPITAL
        self.target_profit_usd = self.virtual_capital * (Config.TARGET_PROFIT_PCT / 100.0) # $160
        self.max_drawdown_usd = self.virtual_capital * (Config.MAX_DRAWDOWN_PCT / 100.0)   # $160
        self.daily_loss_limit_usd = self.virtual_capital * (Config.MAX_DAILY_LOSS_PCT / 100.0) # $100
        self.circuit_breaker_usd = self.virtual_capital * (Config.DAILY_CIRCUIT_BREAKER_PCT / 100.0) # $70
        self.risk_per_trade_usd = self.virtual_capital * (Config.RISK_PER_TRADE_PCT / 100.0) # $25

        self.state = self._load_state()

    def _get_current_utc_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _default_state(self) -> Dict[str, Any]:
        today = self._get_current_utc_date()
        return {
            "challenge_start_date": Config.CHALLENGE_START_DATE,
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
            "status": "ACTIVE", # ACTIVE, PAUSED, TARGET_HIT, DAILY_LIMIT_HIT, MAX_DRAWDOWN_HIT
            "status_message": "Strategy 2 bot active and monitoring market.",
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
                    state = json.load(f)
                    self._check_day_rollover(state)
                    return state
            except Exception as e:
                logger.error(f"Error loading state.json: {e}")

        default_state = self._default_state()
        self._save_state(default_state)
        return default_state

    def _save_state(self, state: Optional[Dict[str, Any]] = None):
        """Save state to state.json."""
        if state is None:
            state = self.state
        state["last_update"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write state.json: {e}")

    def _check_day_rollover(self, state: Dict[str, Any]):
        """Reset daily stats at 00:00 UTC."""
        today = self._get_current_utc_date()
        if state.get("current_day_utc") != today:
            logger.info(f"Daily Rollover detected (Old: {state.get('current_day_utc')}, New: {today}). Resetting daily stats.")
            current_equity = self.virtual_capital + state.get("total_realized_pl", 0.0)
            state["current_day_utc"] = today
            state["starting_balance_today"] = current_equity
            state["daily_realized_pl"] = 0.0
            state["daily_trades_count"] = 0
            state["daily_losses_count"] = 0

            if state.get("status") == "DAILY_LIMIT_HIT":
                state["status"] = "ACTIVE"
                state["status_message"] = "New trading day started. Daily circuit breaker reset."

    def record_closed_trade(self, profit_loss: float):
        """Update metrics when a trade closes."""
        self._check_day_rollover(self.state)

        # Update Daily
        self.state["daily_realized_pl"] += profit_loss
        self.state["daily_trades_count"] += 1
        if profit_loss < 0:
            self.state["daily_losses_count"] = self.state.get("daily_losses_count", 0) + 1

        # Update Overall
        self.state["total_realized_pl"] += profit_loss
        self.state["total_trades_count"] += 1
        if profit_loss > 0:
            self.state["total_wins"] += 1
        elif profit_loss < 0:
            self.state["total_losses"] += 1

        current_equity = self.virtual_capital + self.state["total_realized_pl"]
        self.state["equity_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2)
        })

        if len(self.state["equity_history"]) > 100:
            self.state["equity_history"] = self.state["equity_history"][-100:]

        self.update_status(0.0)
        self._save_state()

    def update_status(self, open_unrealized_pl: float = 0.0) -> Dict[str, Any]:
        """Verify prop challenge compliance rules."""
        self._check_day_rollover(self.state)

        total_pl = self.state["total_realized_pl"] + open_unrealized_pl
        daily_pl = self.state["daily_realized_pl"] + open_unrealized_pl

        # 1. Target profit reached (+8%)
        if total_pl >= self.target_profit_usd:
            self.state["status"] = "TARGET_HIT"
            self.state["status_message"] = f"TARGET REACHED! Total Gain: +${total_pl:.2f} (+{(total_pl/self.virtual_capital)*100:.1f}%). Challenge passed!"
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 2. Max Total Drawdown reached (-8%)
        if total_pl <= -self.max_drawdown_usd:
            self.state["status"] = "MAX_DRAWDOWN_HIT"
            self.state["status_message"] = f"MAX DRAWDOWN BREACH! Total Loss: ${total_pl:.2f}. Trading halted permanently."
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 3. Daily Loss Limit breached (-5%)
        if daily_pl <= -self.daily_loss_limit_usd:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"DAILY LIMIT HIT! Loss: ${daily_pl:.2f}. Trading halted until 00:00 UTC."
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 4. Max 2 Daily Losses reached
        if self.state.get("daily_losses_count", 0) >= Config.MAX_DAILY_LOSSES:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = f"MAX DAILY LOSSES HIT ({self.state.get('daily_losses_count')}/{Config.MAX_DAILY_LOSSES}). Circuit breaker triggered until 00:00 UTC."
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 5. Active normal monitoring
        if self.state.get("status") not in ("PAUSED", "TARGET_HIT", "MAX_DRAWDOWN_HIT"):
            self.state["status"] = "ACTIVE"
            self.state["status_message"] = "Strategy 2 active: ALMA + SuperSmoother confluence monitoring."

        self._save_state()
        return self.get_summary(open_unrealized_pl)

    def can_open_trade(self, open_trades_count: int, open_unrealized_pl: float) -> Tuple[bool, str]:
        """Check if trading is permitted."""
        self._check_day_rollover(self.state)

        status = self.state.get("status", "ACTIVE")
        if status == "PAUSED":
            return False, "Bot is manually paused."
        if status == "TARGET_HIT":
            return False, "Challenge target already completed."
        if status == "MAX_DRAWDOWN_HIT":
            return False, "Max drawdown limit reached."
        if status == "DAILY_LIMIT_HIT":
            return False, "Daily circuit breaker active until 00:00 UTC."

        if self.state.get("daily_losses_count", 0) >= Config.MAX_DAILY_LOSSES:
            return False, f"Max daily losses limit reached ({self.state.get('daily_losses_count')}/{Config.MAX_DAILY_LOSSES})."

        if open_trades_count >= Config.MAX_CONCURRENT_TRADES:
            return False, f"Max concurrent trades reached ({open_trades_count}/{Config.MAX_CONCURRENT_TRADES})."

        if self.state.get("daily_trades_count", 0) >= Config.MAX_TRADES_PER_DAY:
            return False, f"Max daily trades limit reached ({self.state['daily_trades_count']}/{Config.MAX_TRADES_PER_DAY})."

        daily_pl = self.state["daily_realized_pl"] + open_unrealized_pl
        remaining_daily_risk = self.circuit_breaker_usd + daily_pl
        if remaining_daily_risk < self.risk_per_trade_usd:
            return False, f"Remaining daily risk budget (${remaining_daily_risk:.2f}) is less than per-trade risk (${self.risk_per_trade_usd:.2f})."

        return True, "Trading allowed."

    def calculate_position_size(self, instrument: str, entry_price: float, stop_loss_price: float) -> int:
        """
        Calculate exact position size so that hitting stop loss
        results in EXACTLY $25 risk (1.25% of $2,000).
        """
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            raise ValueError("Stop loss distance must be greater than zero.")

        risk_usd = self.risk_per_trade_usd # $25.00

        if instrument.endswith("_USD"):
            units = risk_usd / stop_distance
        elif instrument.startswith("USD_"):
            units = (risk_usd * entry_price) / stop_distance
        else:
            units = risk_usd / stop_distance

        units_int = int(round(units))
        units_int = max(100, min(units_int, 50000))
        return units_int

    def get_summary(self, open_unrealized_pl: float = 0.0) -> Dict[str, Any]:
        """Return full metrics for Dashboard API."""
        total_pl = self.state["total_realized_pl"] + open_unrealized_pl
        daily_pl = self.state["daily_realized_pl"] + open_unrealized_pl
        current_equity = self.virtual_capital + total_pl
        total_gain_pct = (total_pl / self.virtual_capital) * 100.0

        overall_profit_usd = max(0.0, total_pl)
        overall_profit_pct = (overall_profit_usd / self.virtual_capital) * 100.0
        profit_remaining_usd = max(0.0, self.target_profit_usd - overall_profit_usd)
        target_progress_pct = min(100.0, max(0.0, (overall_profit_usd / self.target_profit_usd) * 100.0))

        overall_loss_usd = abs(min(0.0, total_pl))
        overall_loss_pct = (overall_loss_usd / self.virtual_capital) * 100.0
        drawdown_buffer_usd = max(0.0, self.max_drawdown_usd - overall_loss_usd)
        drawdown_used_pct = min(100.0, (overall_loss_usd / self.max_drawdown_usd) * 100.0)
        max_drawdown_floor = self.virtual_capital - self.max_drawdown_usd

        daily_loss_usd = abs(min(0.0, daily_pl))
        daily_loss_pct = (daily_loss_usd / self.virtual_capital) * 100.0

        tot_trades = self.state.get("total_trades_count", 0)
        tot_wins = self.state.get("total_wins", 0)
        tot_losses = self.state.get("total_losses", 0)
        win_rate_pct = round((tot_wins / tot_trades) * 100.0, 1) if tot_trades > 0 else 0.0

        return {
            "virtual_capital": self.virtual_capital,
            "current_equity": round(current_equity, 2),
            "total_pl": round(total_pl, 2),
            "total_gain_pct": round(total_gain_pct, 2),
            "target_profit_usd": self.target_profit_usd,
            "target_profit_pct": Config.TARGET_PROFIT_PCT,
            "overall_profit_usd": round(overall_profit_usd, 2),
            "overall_profit_pct": round(overall_profit_pct, 2),
            "profit_remaining_usd": round(profit_remaining_usd, 2),
            "target_progress_pct": round(target_progress_pct, 1),
            "max_drawdown_usd": self.max_drawdown_usd,
            "max_drawdown_pct": Config.MAX_DRAWDOWN_PCT,
            "overall_loss_usd": round(overall_loss_usd, 2),
            "overall_loss_pct": round(overall_loss_pct, 2),
            "drawdown_buffer_usd": round(drawdown_buffer_usd, 2),
            "drawdown_used_pct": round(drawdown_used_pct, 1),
            "max_drawdown_floor": round(max_drawdown_floor, 2),
            "daily_pl": round(daily_pl, 2),
            "daily_loss_usd": round(daily_loss_usd, 2),
            "daily_loss_pct": round(daily_loss_pct, 2),
            "max_daily_loss_usd": self.daily_loss_limit_usd,
            "circuit_breaker_usd": self.circuit_breaker_usd,
            "risk_per_trade_usd": self.risk_per_trade_usd,
            "daily_trades_count": self.state.get("daily_trades_count", 0),
            "daily_losses_count": self.state.get("daily_losses_count", 0),
            "max_daily_losses": Config.MAX_DAILY_LOSSES,
            "total_trades_count": tot_trades,
            "total_wins": tot_wins,
            "total_losses": tot_losses,
            "win_rate_pct": win_rate_pct,
            "status": self.state.get("status", "ACTIVE"),
            "status_message": self.state.get("status_message", ""),
            "current_day_utc": self.state.get("current_day_utc", ""),
            "equity_history": self.state.get("equity_history", [])[-50:]
        }

