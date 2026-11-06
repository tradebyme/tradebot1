import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import Config
from oanda_api import OandaClient
from risk_manager import RiskManager
from strategy import MomentumPullbackStrategy

logger = logging.getLogger("TradingBot")

class TradingBot:
    """Core Trading Bot Engine coordinating Market Data, Strategy, and Risk Management."""

    def __init__(self):
        self.client = OandaClient()
        self.risk_manager = RiskManager()
        self.strategy = MomentumPullbackStrategy()
        
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.known_closed_trade_ids = set()
        self.log_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        # Seed known closed trades so we don't double count historical trades
        self._seed_closed_trades()
        self.log("INFO", "Trading Bot initialized successfully.")

    def log(self, level: str, message: str):
        """Append log to in-memory history for Web UI feed and system logger."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        entry = {"time": timestamp, "level": level, "message": message}
        
        with self._lock:
            self.log_history.append(entry)
            if len(self.log_history) > 200:
                self.log_history = self.log_history[-200:]
        
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)

    def _seed_closed_trades(self):
        """
        Record existing closed trades on startup and synchronize challenge performance directly
        from OANDA broker history so that Render restarts never lose past challenge trades.
        """
        try:
            closed = self.client.get_closed_trades(count=50)
            challenge_start = getattr(Config, "CHALLENGE_START_DATE", "2026-11-05")
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Check if RiskManager needs historical synchronization (e.g. fresh Render container)
            needs_sync = (self.risk_manager.state.get("total_trades_count", 0) == 0)

            if needs_sync and closed:
                self.log("INFO", f"Synchronizing challenge trades from OANDA broker history (since {challenge_start})...")
                total_pl = 0.0
                total_trades = 0
                total_wins = 0
                total_losses = 0
                daily_pl = 0.0
                daily_trades = 0
                daily_losses = 0
                running_equity = self.risk_manager.virtual_capital
                equity_history = [
                    {"time": f"{challenge_start}T00:00:00+00:00", "equity": running_equity}
                ]

                # Process trades in chronological order (oldest to newest)
                sorted_closed = sorted(closed, key=lambda x: x.get("closeTime", ""))

                for trade in sorted_closed:
                    close_time = trade.get("closeTime", "")
                    if close_time >= challenge_start:
                        pl = float(trade.get("realizedPL", 0.0))
                        trade_date = close_time[:10]
                        total_pl += pl
                        total_trades += 1
                        if pl > 0:
                            total_wins += 1
                        elif pl < 0:
                            total_losses += 1

                        running_equity += pl
                        equity_history.append({"time": close_time, "equity": round(running_equity, 2)})

                        # Only attribute to daily PnL if closed today
                        if trade_date == today_utc:
                            daily_pl += pl
                            daily_trades += 1
                            if pl < 0:
                                daily_losses += 1

                self.risk_manager.state["challenge_start_date"] = challenge_start
                self.risk_manager.state["total_realized_pl"] = round(total_pl, 4)
                self.risk_manager.state["total_trades_count"] = total_trades
                self.risk_manager.state["total_wins"] = total_wins
                self.risk_manager.state["total_losses"] = total_losses
                self.risk_manager.state["daily_realized_pl"] = round(daily_pl, 4)
                self.risk_manager.state["daily_trades_count"] = daily_trades
                self.risk_manager.state["daily_losses_count"] = daily_losses
                self.risk_manager.state["equity_history"] = equity_history
                self.risk_manager._save_state()
                self.log("INFO", f"Broker Sync Complete: {total_trades} challenge trades loaded. Total Realized PnL: ${total_pl:+.2f}")

            # Register all closed trade IDs so we don't double count
            for t in closed:
                self.known_closed_trade_ids.add(t["id"])
            self.log("INFO", f"Seeded {len(self.known_closed_trade_ids)} past closed trades from OANDA.")
        except Exception as e:
            self.log("WARNING", f"Could not sync closed trades: {e}")

    def sync_closed_trades(self):
        """Detect newly closed trades and update RiskManager PnL."""
        try:
            recent_closed = self.client.get_closed_trades(count=15)
            for trade in recent_closed:
                trade_id = trade["id"]
                if trade_id not in self.known_closed_trade_ids:
                    self.known_closed_trade_ids.add(trade_id)
                    realized_pl = float(trade.get("realizedPL", 0.0))
                    inst = trade.get("instrument", "")
                    self.log("INFO", f"Trade #{trade_id} on {inst} CLOSED with PnL: ${realized_pl:+.2f}")
                    self.risk_manager.record_closed_trade(realized_pl)
        except Exception as e:
            self.log("ERROR", f"Error syncing closed trades: {e}")

    def scan_once(self) -> Dict[str, Any]:
        """Perform a single market scan, risk check, and order execution pass."""
        try:
            # 1. Fetch account info
            account = self.client.get_account_summary()
            open_trades = self.client.get_open_trades()

            # Calculate open unrealized PnL
            open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)

            # 2. Sync any newly closed trades
            self.sync_closed_trades()

            # 3. Update Risk & Challenge compliance
            risk_summary = self.risk_manager.update_status(open_unrealized_pl)

            # Check if trading is allowed by Risk Manager
            can_trade, reason = self.risk_manager.can_open_trade(len(open_trades), open_unrealized_pl)

            analysis_results = []

            # 4. If trading is permitted, scan instruments
            if can_trade:
                pricing = self.client.get_pricing(Config.INSTRUMENTS)

                for inst in Config.INSTRUMENTS:
                    inst_pricing = pricing.get(inst, {})
                    candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=120)
                    analysis = self.strategy.analyze(inst, candles, inst_pricing)
                    analysis_results.append(analysis)

                    if analysis.get("has_signal"):
                        direction = analysis["direction"]
                        entry = analysis["entry_price"]
                        sl = analysis["stop_loss"]
                        tp = analysis["take_profit"]

                        # Position Sizing
                        units = self.risk_manager.calculate_position_size(inst, entry, sl)
                        order_units = units if direction == "BUY" else -units

                        self.log("INFO", f"SIGNAL TRIGGERED on {inst}: {direction} @ {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Units: {order_units}")

                        # Execute Order
                        order_res = self.client.place_market_order(inst, order_units, sl, tp)
                        if order_res.get("success"):
                            trade_id = order_res.get("trade_id")
                            fill_price = order_res.get("price")
                            self.log("INFO", f"ORDER SUCCESS: {inst} #{trade_id} filled @ {fill_price}")
                            # Stop scanning other instruments once position is opened
                            break
                        else:
                            self.log("ERROR", f"Order execution failed on {inst}: {order_res.get('error')}")
            else:
                self.log("INFO", f"Scan: New entries halted. {reason}")

            return {
                "account": account,
                "open_trades": open_trades,
                "risk_summary": risk_summary,
                "can_trade": can_trade,
                "reason": reason,
                "analysis_results": analysis_results
            }

        except Exception as e:
            self.log("ERROR", f"Error in scan_once: {e}")
            return {"error": str(e)}

    def run_loop(self):
        """Continuous background trading loop."""
        self.is_running = True
        self.log("INFO", f"Trading Bot loop started (polling every {Config.POLL_INTERVAL_SECONDS}s)...")

        while self.is_running:
            try:
                self.scan_once()
            except Exception as e:
                self.log("ERROR", f"Unhandled exception in bot loop: {e}")

            # Sleep in 1-second chunks for quick shutdown
            for _ in range(Config.POLL_INTERVAL_SECONDS):
                if not self.is_running:
                    break
                time.sleep(1)

        self.log("INFO", "Trading Bot loop stopped.")

    def start_background(self):
        """Start the trading bot in a separate background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.info("Bot thread is already running.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self.run_loop, daemon=True, name="TradingBotThread")
        self._thread.start()
        logger.info("Bot background thread launched.")

    def stop_background(self):
        """Stop the background bot thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3)
            logger.info("Bot background thread joined.")

    def emergency_close_all(self) -> List[Dict[str, Any]]:
        """Emergency trigger: Close all open trades on OANDA immediately."""
        self.log("WARNING", "EMERGENCY CLOSE ALL TRIGGERED BY USER!")
        results = self.client.close_all_trades()
        self.sync_closed_trades()
        return results

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate all live data required by the Web Dashboard UI."""
        account = self.client.get_account_summary()
        open_trades = self.client.get_open_trades()
        closed_trades = self.client.get_closed_trades(count=20)
        
        open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)
        risk_summary = self.risk_manager.get_summary(open_unrealized_pl)

        # Fetch live pricing for all configured instruments
        try:
            live_prices = self.client.get_pricing(Config.INSTRUMENTS)
        except Exception:
            live_prices = {}

        # Enrich open trades with live pricing and pips
        enriched_open_trades = []
        if open_trades:
            for t in open_trades:
                inst = t.get("instrument", "")
                open_p = float(t.get("price", 0.0))
                units = int(t.get("currentUnits", 0))
                pip_size = 0.01 if "JPY" in inst else 0.0001
                
                inst_price = live_prices.get(inst, {})
                curr_p = inst_price.get("bid" if units > 0 else "ask", open_p)
                pips = round(((curr_p - open_p) / pip_size) if units > 0 else ((open_p - curr_p) / pip_size), 1)

                sl_price = t.get("stopLossOrder", {}).get("price") if t.get("stopLossOrder") else None
                tp_price = t.get("takeProfitOrder", {}).get("price") if t.get("takeProfitOrder") else None

                enriched_open_trades.append({
                    "id": t.get("id"),
                    "instrument": inst,
                    "units": units,
                    "direction": "BUY" if units > 0 else "SELL",
                    "open_price": open_p,
                    "current_price": curr_p,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
                    "pips": pips,
                    "open_time": t.get("openTime", "")
                })

        # Enrich closed trades
        enriched_closed_trades = []
        for ct in closed_trades:
            pl = float(ct.get("realizedPL", 0.0))
            units = int(ct.get("initialUnits", 0))
            enriched_closed_trades.append({
                "id": ct.get("id"),
                "instrument": ct.get("instrument", ""),
                "direction": "BUY" if units > 0 else "SELL",
                "units": abs(units),
                "open_price": float(ct.get("price", 0.0)),
                "close_price": float(ct.get("averageClosePrice", 0.0)),
                "realized_pl": pl,
                "result": "WIN" if pl > 0 else "LOSS" if pl < 0 else "BE",
                "close_time": ct.get("closeTime", "")
            })

        # Market & Session info
        market_open, market_msg = self.strategy.is_market_open()
        session_active, session_msg = self.strategy.is_session_active()

        with self._lock:
            logs = list(self.log_history[-60:])

        return {
            "account": {
                "id": account.get("id"),
                "currency": account.get("currency", "USD"),
                "balance": float(account.get("balance", 0.0)),
                "nav": float(account.get("NAV", 0.0)),
                "unrealized_pl": float(account.get("unrealizedPL", 0.0)),
                "margin_used": float(account.get("marginUsed", 0.0)),
                "margin_available": float(account.get("marginAvailable", 0.0)),
            },
            "risk": risk_summary,
            "open_trades": enriched_open_trades,
            "closed_trades": enriched_closed_trades,
            "market_status": {
                "market_open": market_open,
                "market_msg": market_msg,
                "session_active": session_active,
                "session_msg": session_msg,
                "instruments": Config.INSTRUMENTS,
                "timeframe": Config.TIMEFRAME
            },
            "live_prices": live_prices,
            "logs": logs,
            "bot_running": self.is_running
        }

