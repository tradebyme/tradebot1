import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from config import Config

logger = logging.getLogger("RiskManager4")


class RiskManager:
    """
    Manages Prop Firm Challenge compliance, Capital Allocation, and Position Sizing.
    Guarantees that maximum loss per trade never exceeds the risk budget ($25 on $2,000 capital).
    """

    def __init__(self, state_file_path: Optional[str] = None):
        if state_file_path is None:
            self.state_file = Path(__file__).resolve().parent / "state.json"
        else:
            self.state_file = Path(state_file_path)

        self.virtual_capital = Config.VIRTUAL_CAPITAL
        self.target_profit_usd = self.virtual_capital * (Config.TARGET_PROFIT_PCT / 100.0)    # $160
        self.max_drawdown_usd = self.virtual_capital * (Config.MAX_DRAWDOWN_PCT / 100.0)      # $160
        self.daily_loss_limit_usd = self.virtual_capital * (Config.MAX_DAILY_LOSS_PCT / 100.0)  # $100
        self.circuit_breaker_usd = self.virtual_capital * (Config.DAILY_CIRCUIT_BREAKER_PCT / 100.0)  # $70
        self.risk_per_trade_usd = self.virtual_capital * (Config.RISK_PER_TRADE_PCT / 100.0)  # $25

        self.state = self._load_state()

    def _get_current_utc_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _default_state(self) -> Dict[str, Any]:
        today = self._get_current_utc_date()
        return {
            "challenge_start_date": today,
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
            "status": "ACTIVE",  # ACTIVE, PAUSED, TARGET_HIT, DAILY_LIMIT_HIT, MAX_DRAWDOWN_HIT
            "status_message": "Bot active and monitoring market.",
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
                logger.error(f"Error loading state.json, initializing fresh state: {e}")

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
            logger.info(
                f"Daily Rollover detected (Old: {state.get('current_day_utc')}, New: {today}). Resetting daily stats."
            )
            current_equity = self.virtual_capital + state.get("total_realized_pl", 0.0)
            state["current_day_utc"] = today
            state["starting_balance_today"] = current_equity
            state["daily_realized_pl"] = 0.0
            state["daily_trades_count"] = 0
            state["daily_losses_count"] = 0

            # Reset status if it was locked solely due to daily limit
            if state.get("status") == "DAILY_LIMIT_HIT":
                state["status"] = "ACTIVE"
                state["status_message"] = "New trading day started. Circuit breaker reset."

    def record_closed_trade(self, profit_loss: float):
        """Update metrics when a trade closes."""
        self._check_day_rollover(self.state)
        self.state["daily_realized_pl"] += profit_loss
        self.state["total_realized_pl"] += profit_loss
        self.state["daily_trades_count"] += 1
        self.state["total_trades_count"] += 1

        if profit_loss < 0:
            self.state["daily_losses_count"] = self.state.get("daily_losses_count", 0) + 1
            self.state["total_losses"] = self.state.get("total_losses", 0) + 1
        elif profit_loss > 0:
            self.state["total_wins"] = self.state.get("total_wins", 0) + 1

        current_equity = self.virtual_capital + self.state["total_realized_pl"]
        self.state["equity_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(current_equity, 2)
        })
        if len(self.state["equity_history"]) > 500:
            self.state["equity_history"] = self.state["equity_history"][-500:]

        logger.info(
            f"Recorded closed trade PnL: ${profit_loss:+.2f}. "
            f"Daily PnL: ${self.state['daily_realized_pl']:+.2f}, "
            f"Total Realized: ${self.state['total_realized_pl']:+.2f}"
        )
        self._save_state()

    def update_status(self, open_unrealized_pl: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate compliance against:
        1. 8% Target Profit ($160 gain)
        2. 5% Daily Loss Limit ($100 max daily loss, circuit breaker at -$70)
        3. 8% Max Drawdown ($160 max loss from starting capital)
        """
        self._check_day_rollover(self.state)

        total_pl = self.state["total_realized_pl"] + open_unrealized_pl
        daily_pl = self.state["daily_realized_pl"] + open_unrealized_pl

        # 1. Target Check (+8%)
        if total_pl >= self.target_profit_usd:
            self.state["status"] = "TARGET_HIT"
            self.state["status_message"] = (
                f"TARGET ACHIEVED! Profit is ${total_pl:.2f} "
                f"(+{(total_pl / self.virtual_capital) * 100:.1f}%). Trading halted to lock gains."
            )
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 2. Max Total Drawdown Check (-8%)
        if total_pl <= -self.max_drawdown_usd:
            self.state["status"] = "MAX_DRAWDOWN_HIT"
            self.state["status_message"] = (
                f"MAX DRAWDOWN HIT: Loss is ${abs(total_pl):.2f} (8% limit). Trading halted to protect capital."
            )
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # 3. Daily Loss Circuit Breaker Check (-3.5% or 2 consecutive losses)
        if daily_pl <= -self.circuit_breaker_usd:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = (
                f"DAILY CIRCUIT BREAKER: Daily loss reached ${abs(daily_pl):.2f} "
                f"(-{Config.DAILY_CIRCUIT_BREAKER_PCT}% limit). Paused until 00:00 UTC."
            )
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        if self.state.get("daily_losses_count", 0) >= Config.MAX_DAILY_LOSSES:
            self.state["status"] = "DAILY_LIMIT_HIT"
            self.state["status_message"] = (
                f"DAILY CIRCUIT BREAKER: Hit {Config.MAX_DAILY_LOSSES} losses today. "
                "Paused until 00:00 UTC to prevent revenge trading."
            )
            self._save_state()
            return self.get_summary(open_unrealized_pl)

        # If previously active, keep active
        if self.state.get("status") in ("ACTIVE", "PAUSED"):
            pass
        else:
            self.state["status"] = "ACTIVE"
            self.state["status_message"] = "Bot active and monitoring market."

        self._save_state()
        return self.get_summary(open_unrealized_pl)

    def can_open_trade(self, current_open_trades_count: int, open_unrealized_pl: float = 0.0) -> Tuple[bool, str]:
        """Check all risk & compliance conditions before allowing order placement."""
        self.update_status(open_unrealized_pl)

        status = self.state.get("status")
        if status == "PAUSED":
            return False, "Bot is currently paused by user."
        if status == "TARGET_HIT":
            return False, "Challenge target already achieved! Trading locked to secure pass."
        if status == "MAX_DRAWDOWN_HIT":
            return False, "Max drawdown limit reached. Trading halted."
        if status == "DAILY_LIMIT_HIT":
            return False, "Daily circuit breaker active. Trading halted until 00:00 UTC."

        if current_open_trades_count >= Config.MAX_CONCURRENT_TRADES:
            return False, f"Max concurrent trades reached ({current_open_trades_count}/{Config.MAX_CONCURRENT_TRADES})."

        if self.state.get("daily_trades_count", 0) >= Config.MAX_TRADES_PER_DAY:
            return False, f"Max daily trades reached ({self.state.get('daily_trades_count')}/{Config.MAX_TRADES_PER_DAY})."

        return True, "Trading allowed."

    def calculate_position_size(self, instrument: str, entry_price: float, stop_loss_price: Optional[float] = None) -> int:
        """
        Calculate position units according to Pine Script sizing mode:
        1. "Fixed Units": uses Config.FIXED_QTY (default 10,000 units = 0.10 lot)
        2. "Risk % of Equity": dynamic lot sizing guaranteeing loss <= risk budget ($25 on $2k capital)
        """
        if getattr(Config, "SIZE_MODE", "Risk % of Equity") == "Fixed Units":
            return int(getattr(Config, "FIXED_QTY", 10000.0))

        pip_unit = 0.01 if "JPY" in instrument else (0.10 if "XAU" in instrument else 0.0001)
        if stop_loss_price is not None:
            distance = abs(entry_price - stop_loss_price)
        else:
            distance = 20.0 * pip_unit

        if distance <= 0.00001:
            logger.warning("Stop loss distance too small, using minimum safe distance.")
            distance = 20.0 * pip_unit

        # Determine quote currency conversion
        if "USD" in instrument and instrument.endswith("USD"):
            # Counter currency is USD (e.g. GBP_USD, EUR_USD, AUD_USD, XAU_USD)
            # Loss per unit = distance * 1.0 USD
            units = int(self.risk_per_trade_usd / distance)
        elif instrument.endswith("JPY"):
            # Counter currency is JPY (e.g. USD_JPY, GBP_JPY)
            # pip_value in USD approx = (0.01 / entry_price) * units
            loss_per_unit = distance / entry_price
            units = int(self.risk_per_trade_usd / max(loss_per_unit, 1e-6))
        else:
            # Conservative standard approximation for cross pairs
            units = int(self.risk_per_trade_usd / distance)

        # Clamping position size to safe prop bounds
        if "XAU" in instrument:
            # Gold: 1 unit = 1 oz. Clamp between 1 and 20 units
            units = max(1, min(units, 20))
        else:
            # Standard Forex: Clamp between 1,000 units (micro-lot) and 50,000 units (half-lot)
            units = max(1000, min(units, 50000))

        logger.info(
            f"Calculated Position Size for {instrument}: {units} units "
            f"(Risk: ${self.risk_per_trade_usd:.2f}, SL Distance: {distance:.5f})"
        )
        return units

    def get_summary(self, open_unrealized_pl: float = 0.0) -> Dict[str, Any]:
        """Generate high-level compliance summary for the dashboard."""
        total_pl = self.state.get("total_realized_pl", 0.0) + open_unrealized_pl
        daily_pl = self.state.get("daily_realized_pl", 0.0) + open_unrealized_pl
        current_equity = self.virtual_capital + total_pl

        progress_to_target_pct = min(100.0, max(0.0, (total_pl / self.target_profit_usd) * 100.0))
        drawdown_used_pct = min(100.0, max(0.0, (abs(min(0.0, total_pl)) / self.max_drawdown_usd) * 100.0))
        daily_loss_used_pct = min(100.0, max(0.0, (abs(min(0.0, daily_pl)) / self.daily_loss_limit_usd) * 100.0))

        total_trades = self.state.get("total_trades_count", 0)
        total_wins = self.state.get("total_wins", 0)
        win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "virtual_capital": self.virtual_capital,
            "current_equity": round(current_equity, 2),
            "target_profit_usd": self.target_profit_usd,
            "max_drawdown_usd": self.max_drawdown_usd,
            "daily_loss_limit_usd": self.daily_loss_limit_usd,
            "circuit_breaker_usd": self.circuit_breaker_usd,
            "total_pl": round(total_pl, 2),
            "daily_pl": round(daily_pl, 2),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": self.state.get("total_losses", 0),
            "daily_trades": self.state.get("daily_trades_count", 0),
            "daily_losses": self.state.get("daily_losses_count", 0),
            "win_rate": round(win_rate, 1),
            "progress_to_target_pct": round(progress_to_target_pct, 1),
            "drawdown_used_pct": round(drawdown_used_pct, 1),
            "daily_loss_used_pct": round(daily_loss_used_pct, 1),
            "status": self.state.get("status", "ACTIVE"),
            "status_message": self.state.get("status_message", "Bot active.")
        }

    def set_pause(self, paused: bool):
        """Manually pause or resume the bot."""
        self.state["status"] = "PAUSED" if paused else "ACTIVE"
        self.state["status_message"] = "Bot paused by user." if paused else "Bot resumed by user."
        self._save_state()

    def reset_challenge(self):
        """Reset the challenge state back to the initial baseline."""
        self.state = self._default_state()
        self._save_state()
        logger.info("Challenge state has been reset to baseline $2,000.")

