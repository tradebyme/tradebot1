import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import Config
from oanda_api import OandaClient
from risk_manager import RiskManager
from strategy import ConfluenceTrendStrategy

logger = logging.getLogger("TradingBot2")

class TradingBot:
    """Core Trading Bot Engine for Strategy 2: Adaptive SuperSmoother + Trend Target Ribbon."""

    def __init__(self):
        self.client = OandaClient()
        self.risk_manager = RiskManager()
        self.strategy = ConfluenceTrendStrategy()

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.known_closed_trade_ids = set()
        self.log_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        # Seed and sync closed trades from OANDA
        self._seed_closed_trades()
        self.log("INFO", "Strategy 2 Trading Bot initialized successfully.")

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
        """Sync closed trades from broker so restarts maintain accurate P&L."""
        try:
            closed = self.client.get_closed_trades(count=50)
            challenge_start = Config.CHALLENGE_START_DATE
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            needs_sync = (self.risk_manager.state.get("total_trades_count", 0) == 0)

            if needs_sync and closed:
                self.log("INFO", f"Synchronizing challenge trades from OANDA (since {challenge_start})...")
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

                        if trade_date == today_utc:
                            daily_pl += pl
                            daily_trades += 1
                            if pl < 0:
                                daily_losses += 1

                self.risk_manager.state["total_realized_pl"] = round(total_pl, 4)
                self.risk_manager.state["total_trades_count"] = total_trades
                self.risk_manager.state["total_wins"] = total_wins
                self.risk_manager.state["total_losses"] = total_losses
                self.risk_manager.state["daily_realized_pl"] = round(daily_pl, 4)
                self.risk_manager.state["daily_trades_count"] = daily_trades
                self.risk_manager.state["daily_losses_count"] = daily_losses
                self.risk_manager.state["equity_history"] = equity_history
                self.risk_manager._save_state()
                self.log("INFO", f"Broker Sync Complete: {total_trades} challenge trades loaded. Total PnL: ${total_pl:+.2f}")

            for t in closed:
                self.known_closed_trade_ids.add(t["id"])
            self.log("INFO", f"Seeded {len(self.known_closed_trade_ids)} past closed trades from broker.")
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
            account = self.client.get_account_summary()
            open_trades = self.client.get_open_trades()
            open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)

            self.sync_closed_trades()
            self.risk_manager.update_status(open_unrealized_pl)

            can_trade, reason = self.risk_manager.can_open_trade(len(open_trades), open_unrealized_pl)
            analysis_results = []

            if can_trade:
                pricing = self.client.get_pricing(Config.INSTRUMENTS)

                for inst in Config.INSTRUMENTS:
                    inst_pricing = pricing.get(inst, {})
                    candles = self.client.get_candles(inst, granularity=Config.TIMEFRAME, count=120)

                    if not candles:
                        continue

                    spread_pips = inst_pricing.get("spread_pips", 1.0)
                    analysis = self.strategy.analyze(candles, current_spread_pips=spread_pips)
                    analysis_results.append({"instrument": inst, "analysis": analysis})

                    signal = analysis.get("signal")
                    if signal in ("BUY", "SELL"):
                        entry_price = analysis["entry_price"]
                        stop_loss = analysis["stop_loss"]
                        take_profit = analysis["take_profit"]

                        units = self.risk_manager.calculate_position_size(inst, entry_price, stop_loss)
                        if signal == "SELL":
                            units = -units

                        self.log("INFO", f"SIGNAL TRIGGERED on {inst}: {signal} @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Units: {units}")

                        try:
                            order_res = self.client.create_market_order(inst, units, stop_loss, take_profit)
                            trade_id = order_res.get("id")
                            fill_price = order_res.get("price")
                            self.log("INFO", f"ORDER SUCCESS: {inst} #{trade_id} filled @ {fill_price}")
                            break
                        except Exception as oe:
                            self.log("ERROR", f"Failed to execute {signal} on {inst}: {oe}")
            else:
                self.log("INFO", f"Scan: New entries halted. {reason}")

            return {
                "status": "success",
                "can_trade": can_trade,
                "reason": reason,
                "analysis": analysis_results
            }

        except Exception as e:
            self.log("ERROR", f"Error in scan_once: {e}")
            return {"status": "error", "message": str(e)}

    def _run_loop(self):
        """Continuous background execution thread."""
        self.log("INFO", f"Strategy 2 loop started (polling every {Config.POLL_INTERVAL_SECONDS}s)...")
        while self.is_running:
            try:
                self.scan_once()
            except Exception as e:
                self.log("ERROR", f"Loop exception: {e}")
            time.sleep(Config.POLL_INTERVAL_SECONDS)

    def start_background(self):
        """Start the background scanning thread."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self.log("INFO", "Strategy 2 background thread launched.")

    def stop(self):
        """Stop background execution."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.log("INFO", "Trading Bot stopped.")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Compile complete real-time metrics for Dashboard Web UI."""
        account = self.client.get_account_summary()
        open_trades = self.client.get_open_trades()
        closed_trades = self.client.get_closed_trades(count=15)
        live_prices = self.client.get_pricing(Config.INSTRUMENTS)

        open_unrealized_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)
        risk_summary = self.risk_manager.get_summary(open_unrealized_pl)

        enriched_open_trades = []
        for ot in open_trades:
            inst = ot.get("instrument")
            price_info = live_prices.get(inst, {})
            current_price = price_info.get("bid" if int(ot.get("currentUnits", 0)) > 0 else "ask", float(ot.get("price", 0.0)))
            entry_price = float(ot.get("price", 0.0))
            units = int(ot.get("currentUnits", 0))
            is_long = units > 0
            pip_mult = 100.0 if "JPY" in inst else 10000.0
            pips = round((current_price - entry_price) * pip_mult if is_long else (entry_price - current_price) * pip_mult, 1)

            enriched_open_trades.append({
                "id": ot.get("id"),
                "instrument": inst,
                "direction": "BUY" if is_long else "SELL",
                "units": abs(units),
                "open_price": entry_price,
                "current_price": current_price,
                "stop_loss": float(ot.get("stopLossOrder", {}).get("price", 0.0)),
                "take_profit": float(ot.get("takeProfitOrder", {}).get("price", 0.0)),
                "unrealized_pl": float(ot.get("unrealizedPL", 0.0)),
                "pips": pips,
                "open_time": ot.get("openTime")
            })

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
