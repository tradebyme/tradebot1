import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List

from config import Config
from strategy import LiquiditySweepStrategy
from risk_manager import RiskManager
from oanda_api import OandaAPI

logger = logging.getLogger("SMC_Bot")

class TradingBot:
    """Automated Trading Bot for Strategy 3 (ICT / SMC Liquidity Sweep)."""

    def __init__(self):
        self.api = OandaAPI()
        self.risk_manager = RiskManager()
        self.strategies = {inst: LiquiditySweepStrategy(inst) for inst in Config.INSTRUMENTS}
        self.is_running = False
        self.worker_thread = None
        self.known_closed_trades = set()
        self.bot_status = {
            "status": "INITIALIZING",
            "active_trade": None,
            "last_check_time": None,
            "monitored_instruments": Config.INSTRUMENTS,
            "recent_signals": {},
            "active_traded_instrument": Config.INSTRUMENTS[0]
        }

    def sync_broker_history(self):
        """Synchronize recently closed broker trades on startup."""
        logger.info("Synchronizing closed trades from OANDA broker history...")
        trades = self.api.get_closed_trades(count=20)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for tr in trades:
            t_id = tr.get("id")
            if t_id and t_id not in self.known_closed_trades:
                self.known_closed_trades.add(t_id)
                close_time_str = tr.get("closeTime", "")
                realized_pnl = float(tr.get("realizedPL", 0.0))
                instrument = tr.get("instrument", "")
                initial_units = float(tr.get("initialUnits", 0.0))
                direction = "BUY" if initial_units > 0 else "SELL"

                if close_time_str.startswith(today_str):
                    logger.info(f"Loaded today's broker trade #{t_id} ({instrument} {direction}): PnL ${realized_pnl:+.2f}")

    def check_open_positions(self):
        """Track open positions and detect when an active trade closes."""
        open_trades = self.api.get_open_trades()

        if open_trades:
            tr = open_trades[0]
            inst = tr.get("instrument", Config.INSTRUMENTS[0])
            self.bot_status["active_traded_instrument"] = inst
            self.bot_status["active_trade"] = {
                "id": tr.get("id"),
                "instrument": inst,
                "units": tr.get("currentUnits"),
                "price": float(tr.get("price", 0.0)),
                "unrealized_pnl": float(tr.get("unrealizedPL", 0.0)),
                "stop_loss": float(tr.get("stopLossOrder", {}).get("price", 0.0)) if tr.get("stopLossOrder") else 0.0,
                "take_profit": float(tr.get("takeProfitOrder", {}).get("price", 0.0)) if tr.get("takeProfitOrder") else 0.0,
                "open_time": tr.get("openTime")
            }
        else:
            # If an active trade was tracked, check if it just closed
            if self.bot_status["active_trade"] is not None:
                closed = self.api.get_closed_trades(count=5)
                for c in closed:
                    c_id = c.get("id")
                    if c_id not in self.known_closed_trades:
                        self.known_closed_trades.add(c_id)
                        pnl = float(c.get("realizedPL", 0.0))
                        logger.info(f"Active trade #{c_id} closed with PnL: ${pnl:+.2f}")
                        self.risk_manager.record_closed_trade({
                            "instrument": c.get("instrument"),
                            "realized_pnl": pnl,
                            "direction": "BUY" if float(c.get("initialUnits", 0)) > 0 else "SELL"
                        })
                self.bot_status["active_trade"] = None

    def scan_and_execute(self):
        """Scan each instrument for Asian Liquidity Sweeps in London Killzone."""
        self.bot_status["last_check_time"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # Check existing open position
        if self.bot_status["active_trade"] is not None:
            self.bot_status["status"] = "POSITION_ACTIVE"
            return

        # Check risk allowance
        can_trade, risk_msg = self.risk_manager.can_open_trade()
        if not can_trade:
            self.bot_status["status"] = "HALTED"
            self.bot_status["status_reason"] = risk_msg
            return

        self.bot_status["status"] = "SCANNING"

        for instrument, strat in self.strategies.items():
            pricing = self.api.get_pricing(instrument)
            if not pricing:
                continue

            spread = pricing["spread_pips"]
            candles = self.api.get_candles(instrument, count=40, granularity=Config.TIMEFRAME)

            signal_data = strat.analyze(candles, spread)
            self.bot_status["recent_signals"][instrument] = {
                "time": self.bot_status["last_check_time"],
                "signal": signal_data["signal"],
                "reason": signal_data["reason"],
                "spread": spread,
                "asian_range": f"{signal_data['asian_low']} - {signal_data['asian_high']} ({signal_data['asian_range_pips']} pips)"
            }

            if signal_data["signal"] in ("BUY", "SELL"):
                logger.info(f"[{instrument}] SIGNAL TRIGGERED: {signal_data['signal']} at {signal_data['entry_price']}")
                units = self.risk_manager.calculate_position_size(
                    instrument, signal_data["risk_pips"], signal_data["entry_price"]
                )
                if signal_data["signal"] == "SELL":
                    units = -units

                order_res = self.api.create_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=signal_data["stop_loss"],
                    take_profit=signal_data["take_profit"]
                )

                if order_res:
                    self.bot_status["active_traded_instrument"] = instrument
                    self.check_open_positions()
                    break

    def run_loop(self):
        """Main loop execution thread."""
        logger.info("Strategy 3 Trading Bot loop started.")
        self.sync_broker_history()

        while self.is_running:
            try:
                self.check_open_positions()
                self.scan_and_execute()
            except Exception as e:
                logger.error(f"Error in main bot loop: {e}", exc_info=True)
            time.sleep(Config.POLL_INTERVAL_SECONDS)

    def start(self):
        """Start trading bot background worker."""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self.run_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Strategy 3 Trading Bot daemon thread launched.")

    def stop(self):
        """Stop background worker."""
        self.is_running = False
        logger.info("Strategy 3 Trading Bot stopped.")
