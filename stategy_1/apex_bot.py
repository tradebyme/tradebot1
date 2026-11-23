import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import Config
from oanda_api import OandaClient
from apex_risk_manager import ApexRiskManager
from apex_strategy import ApexTrendEMAATRStrategy

logger = logging.getLogger("ApexTradingBot")


def is_apex_trade(trade: Dict[str, Any]) -> bool:
    """Strictly isolate APEX SNIPER trades from Strategy 1 and retail arena bots."""
    ce = trade.get("clientExtensions") or {}
    tag = str(ce.get("tag", "")).upper()
    comment = str(ce.get("comment", "")).upper()
    ext_id = str(ce.get("id", "")).upper()

    # Reject Strategy 1 and retail bots
    if "STRAT_1" in tag or "STRAT_1" in comment or "STRAT_1" in ext_id:
        return False
    if any(k in tag or k in comment for k in ["TITAN", "PHANTOM", "NEXUS", "PREDATOR"]):
        return False

    # Accept if tagged APEX or if on apex instruments
    if "APEX" in tag or "APEX" in comment or "APEX" in ext_id:
        return True

    inst = trade.get("instrument", "")
    if inst == "XAU_USD":
        return True

    return False


class ApexTradingBot:
    """
    APEX SNIPER / Strategy 5: Institutional Trend-ATR Portfolio Bot Engine.
    Autonomous multi-asset coordinator running in background thread alongside Strategy 1.
    """

    def __init__(self):
        apex_api_key = os.getenv("OANDA_API_KEY_APEX") or os.getenv("OANDA_API_KEY_5") or Config.API_KEY
        apex_account_id = os.getenv("OANDA_ACCOUNT_ID_APEX") or os.getenv("OANDA_ACCOUNT_ID_5") or Config.ACCOUNT_ID

        self.client = OandaClient(api_key=apex_api_key, account_id=apex_account_id)
        self.risk_manager = ApexRiskManager()
        self.strategy = ApexTrendEMAATRStrategy()

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.known_closed_trade_ids = set()
        self.log_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        # Dynamic Peak Trailing State: {trade_id: {"instrument": str, "direction": str, "entry": float, "peak": float, "atr": float, "current_sl": float}}
        self.trailing_states: Dict[str, Dict[str, Any]] = {}

        self.instruments = [i.strip() for i in os.getenv("APEX_INSTRUMENTS", "XAU_USD,EUR_USD,GBP_USD").split(",")]
        self.tf_map = {
            "XAU_USD": os.getenv("TF_XAUUSD", "H4"),
            "EUR_USD": os.getenv("TF_EURUSD", "H4"),
            "GBP_USD": os.getenv("TF_GBPUSD", "H1"),
        }

        self._seed_closed_trades()
        self.log("INFO", "APEX SNIPER Institutional Engine initialized successfully.")

    def log(self, level: str, message: str):
        """Append log entry for Web UI feed and console logger."""
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
        """Record existing closed trades and sync realized P&L from OANDA on startup."""
        try:
            summary = self.client.get_account_summary()
            acct = summary.get("account", {})
            broker_pl = float(acct.get("resettablePL", 0.0))
            broker_bal = float(acct.get("balance", self.risk_manager.virtual_capital))

            # Auto-sync baseline if state was wiped by container restart
            if self.risk_manager.state.get("total_trades_count", 0) == 0 and broker_pl != 0.0:
                self.risk_manager.state["total_realized_pl"] = round(broker_pl, 2)
                self.risk_manager._save_state(self.risk_manager.state)
                self.log("INFO", f"Synced broker realized P&L on restart: ${broker_pl:.2f}")

            raw_closed = self.client.get_closed_trades(count=50)
            closed = [t for t in raw_closed if is_apex_trade(t)]
            for t in closed:
                tid = t.get("id")
                if tid:
                    self.known_closed_trade_ids.add(tid)
            self.log("INFO", f"Seeded {len(self.known_closed_trade_ids)} historical APEX trades from OANDA (Account: {self.client.account_id}).")
        except Exception as e:
            self.log("WARNING", f"Could not seed closed trades: {e}")

    def start_background(self):
        """Launch autonomous trading loop in background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.log("INFO", "APEX SNIPER trading loop started in background.")

    def stop_background(self):
        self.is_running = False
        self.log("INFO", "APEX SNIPER trading loop stopped.")

    def _run_loop(self):
        """Core polling loop executing every 25 seconds."""
        while self.is_running:
            try:
                self._check_closed_trades()
                self._update_trailing_stops()
                self._scan_for_signals()
            except Exception as e:
                self.log("ERROR", f"Exception in APEX loop: {e}")

            time.sleep(25)

    def _check_closed_trades(self):
        """Detect any newly closed trades and update risk manager."""
        try:
            raw_closed = self.client.get_closed_trades(count=20)
            recent_closed = [t for t in raw_closed if is_apex_trade(t)]
            for t in recent_closed:
                tid = t.get("id")
                if tid and tid not in self.known_closed_trade_ids:
                    self.known_closed_trade_ids.add(tid)
                    realized_pl = float(t.get("realizedPL", 0.0))
                    instrument = t.get("instrument", "UNKNOWN")
                    self.risk_manager.record_closed_trade(tid, instrument, realized_pl)
                    self.log(
                        "INFO",
                        f"Trade Closed: #{tid} {instrument} | Realized P&L: {'+' if realized_pl >= 0 else ''}${realized_pl:.2f}"
                    )
                    if tid in self.trailing_states:
                        del self.trailing_states[tid]
        except Exception as e:
            self.log("WARNING", f"Error checking closed trades: {e}")

    def _update_trailing_stops(self):
        """Dynamic Peak-ATR Trailing: Ratchet Stop Loss at 3.0x ATR behind peak profit."""
        try:
            open_trades = self.client.get_open_trades()
            current_open_ids = {t["id"] for t in open_trades}

            # Clean stale ids
            for tid in list(self.trailing_states.keys()):
                if tid not in current_open_ids:
                    del self.trailing_states[tid]

            if not open_trades:
                return

            instruments = list({t["instrument"] for t in open_trades})
            pricing = self.client.get_pricing(instruments)

            for t in open_trades:
                tid = t["id"]
                inst = t["instrument"]
                units = int(t.get("currentUnits", 0))
                direction = "BUY" if units > 0 else "SELL"
                entry_price = float(t["price"])

                price_info = pricing.get(inst, {})
                current_price = price_info.get("mid", entry_price)

                # Initialize tracking if new
                if tid not in self.trailing_states:
                    sl_obj = t.get("stopLossOrder")
                    sl_price = float(sl_obj["price"]) if sl_obj else (entry_price * 0.98 if direction == "BUY" else entry_price * 1.02)
                    approx_atr = abs(entry_price - sl_price) / 2.0 if sl_price > 0 else (entry_price * 0.005)

                    self.trailing_states[tid] = {
                        "instrument": inst,
                        "direction": direction,
                        "entry_price": entry_price,
                        "peak_price": entry_price,
                        "atr": approx_atr,
                        "current_sl": sl_price
                    }

                state = self.trailing_states[tid]
                atr = state["atr"]
                curr_sl = state["current_sl"]

                if direction == "BUY":
                    if current_price > state["peak_price"]:
                        state["peak_price"] = current_price

                    if (state["peak_price"] - entry_price) >= (1.0 * atr):
                        new_sl = state["peak_price"] - (3.0 * atr)
                        if new_sl > curr_sl:
                            self.client.modify_trade_stop_loss(tid, new_sl, inst)
                            self.log("INFO", f"Trailing SL Ratchet: #{tid} {inst} moved up {curr_sl:.4f} -> {new_sl:.4f}")
                            state["current_sl"] = new_sl

                elif direction == "SELL":
                    if current_price < state["peak_price"]:
                        state["peak_price"] = current_price

                    if (entry_price - state["peak_price"]) >= (1.0 * atr):
                        new_sl = state["peak_price"] + (3.0 * atr)
                        if new_sl < curr_sl or curr_sl <= 0:
                            self.client.modify_trade_stop_loss(tid, new_sl, inst)
                            self.log("INFO", f"Trailing SL Ratchet: #{tid} {inst} moved down {curr_sl:.4f} -> {new_sl:.4f}")
                            state["current_sl"] = new_sl

        except Exception as e:
            self.log("WARNING", f"Error in trailing stop update: {e}")

    def _scan_for_signals(self):
        """Scan configured instruments across H4/H1 timeframes for 200 EMA + 20/50 crossovers."""
        try:
            open_trades = self.client.get_open_trades()
            open_count = len(open_trades)
            open_instruments = {t["instrument"] for t in open_trades}

            can_trade, reason = self.risk_manager.can_open_new_trade(
                current_open_count=open_count,
                max_concurrent=int(os.getenv("APEX_MAX_CONCURRENT_TRADES", "2")),
                max_daily_trades=int(os.getenv("APEX_MAX_TRADES_PER_DAY", "3"))
            )
            if not can_trade:
                return

            for inst in self.instruments:
                if inst in open_instruments:
                    continue

                tf = self.tf_map.get(inst, "H4")
                candles = self.client.get_candles(inst, granularity=tf, count=250)
                if not candles:
                    continue

                df = self.strategy.parse_candles(candles)
                eval_res = self.strategy.evaluate_signal(df, inst)
                sig = eval_res.get("signal")

                if sig in ("BUY", "SELL"):
                    entry = eval_res["price"]
                    sl = eval_res["sl"]
                    atr = eval_res["atr"]
                    pip_size, pip_val, _ = self.client.get_pip_specs(inst)

                    units = self.risk_manager.calculate_position_size(entry, sl, pip_val)
                    if sig == "SELL":
                        units = -units

                    order_res = self.client.place_market_order(
                        instrument=inst,
                        units=units,
                        stop_loss_price=sl,
                        take_profit_price=0.0,  # Dynamic ATR trailing used instead of static TP
                        client_tag="APEX_SNIPER"
                    )

                    fill = order_res.get("orderFillTransaction") or order_res.get("raw", {}).get("orderFillTransaction", {})
                    if fill:
                        tid = fill.get("id")
                        self.log("INFO", f"ORDER FILLED: #{tid} {sig} {inst} Units: {units} | SL: {sl:.4f} | ATR: {atr:.4f}")
                        if tid:
                            self.trailing_states[tid] = {
                                "instrument": inst,
                                "direction": sig,
                                "entry_price": entry,
                                "peak_price": entry,
                                "atr": atr,
                                "current_sl": sl
                            }
                    else:
                        self.log("WARNING", f"Order submitted but fill response pending: {order_res}")

        except Exception as e:
            self.log("ERROR", f"Error scanning signals in APEX engine: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate real-time metrics for the APEX SNIPER dashboard (isolated to APEX)."""
        try:
            raw_open = self.client.get_open_trades()
            open_trades = [t for t in raw_open if is_apex_trade(t)]
            floating_pl = sum(float(t.get("unrealizedPL", 0.0)) for t in open_trades)
        except Exception:
            open_trades = []
            floating_pl = 0.0

        try:
            raw_closed = self.client.get_closed_trades(count=30)
            closed_trades = [t for t in raw_closed if is_apex_trade(t)]
        except Exception:
            closed_trades = []

        metrics = self.risk_manager.get_dashboard_metrics(floating_pl)

        with self._lock:
            logs = list(self.log_history)

        return {
            "metrics": metrics,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
            "logs": logs,
            "is_running": (self.risk_manager.state.get("status") == "ACTIVE"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def toggle_pause(self) -> str:
        current_status = self.risk_manager.state.get("status")
        if current_status == "PAUSED":
            self.risk_manager.set_pause(False)
            self.log("INFO", "APEX SNIPER resumed by operator.")
            return "ACTIVE"
        else:
            self.risk_manager.set_pause(True)
            self.log("WARNING", "APEX SNIPER paused by operator.")
            return "PAUSED"

    def emergency_close_all(self) -> List[Dict[str, Any]]:
        self.log("WARNING", "Emergency Close All triggered on APEX SNIPER.")
        results = self.client.close_all_trades()
        self.trailing_states.clear()
        return results

    def reset_challenge(self):
        self.risk_manager.reset_challenge()
        self.trailing_states.clear()
        self.log("WARNING", "APEX SNIPER challenge metrics reset to baseline $2,000.")
